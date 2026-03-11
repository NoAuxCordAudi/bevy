enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::pbr_functions::calculate_tbn_mikktspace
#import bevy_pbr::utils::{rand_f, rand_vec2f, sample_cosine_hemisphere}
#import bevy_render::maths::PI
#import bevy_render::view::View
#import bevy_solari::brdf::{evaluate_brdf, evaluate_brdf_bidirectional, is_specular_surface, BrdfEvalResult}
#import bevy_solari::sampling::{sample_light_emission, sample_ggx_vndf, ggx_vndf_pdf, trace_point_visibility, LightEmissionSample}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, ResolvedRayHitFull, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD}
#import bevy_solari::light_vertex::{LightPathVertex, VcmParams}

/// Maximum number of bounces for a light subpath.
const MAX_LIGHT_BOUNCES = 9u;

/// Fixed-point scale factor for atomic splatting (2^16).
const SPLAT_SCALE: f32 = 65536.0;
const SPLAT_SCALE_INV: f32 = 1.0 / 65536.0;

@group(1) @binding(0) var<storage, read_write> light_vertices: array<LightPathVertex>;
@group(1) @binding(1) var<storage, read_write> light_vertex_count: atomic<u32>;
@group(1) @binding(2) var<uniform> vcm_params: VcmParams;
@group(1) @binding(3) var<uniform> view: View;
@group(1) @binding(4) var<storage, read_write> splat_buffer: array<atomic<u32>>;

/// Checks whether a vec3 contains only finite, non-NaN values within a
/// reasonable magnitude. Used to discard degenerate values that could
/// corrupt buffers or produce fireflies.
fn is_valid(v: vec3<f32>) -> bool {
    return all(v == clamp(v, vec3(-1e20), vec3(1e20)));
}

/// Checks whether a radiance value is valid for accumulation:
/// finite, non-NaN, non-negative, and within a reasonable magnitude.
fn is_valid_radiance(v: vec3<f32>) -> bool {
    return is_valid(v) && all(v >= vec3(0.0));
}

/// Checks whether a scalar MIS-related value is valid (finite, non-NaN,
/// non-negative). Returns 0.0 if invalid.
fn sanitize_mis(v: f32) -> f32 {
    if v == clamp(v, 0.0, 1e20) {
        return v;
    }
    return 0.0;
}

/// Atomically splats a radiance contribution to a pixel in the splat buffer.
///
/// Uses fixed-point encoding: float values are multiplied by SPLAT_SCALE
/// and accumulated via atomicAdd on u32 channels. The camera trace shader
/// reads these back and divides by SPLAT_SCALE to recover float values.
fn splat_add(pixel: vec2<u32>, color: vec3<f32>) {
    let idx = (pixel.y * vcm_params.screen_width + pixel.x) * 3u;
    // Clamp to prevent overflow of u32 atomics. Max value is ~32767 in
    // floating point before scaling, which is plenty for HDR.
    let max_val = f32(0x7FFFFFFFu);
    atomicAdd(&splat_buffer[idx + 0u], u32(clamp(color.r * SPLAT_SCALE, 0.0, max_val)));
    atomicAdd(&splat_buffer[idx + 1u], u32(clamp(color.g * SPLAT_SCALE, 0.0, max_val)));
    atomicAdd(&splat_buffer[idx + 2u], u32(clamp(color.b * SPLAT_SCALE, 0.0, max_val)));
}

/// Connects the current light subpath vertex to the camera (pinhole) and
/// splats the contribution to the appropriate pixel.
///
/// This implements the "t=1" strategy in VCM: the light subpath vertex is
/// directly connected to the camera without any camera-side bounces.
///
/// For a pinhole camera, the "camera PDF" is a Dirac delta in position
/// and a solid-angle density related to the pixel's projected area.
/// Following SmallVCM's convention, the MIS weight simplifies to:
///   mis_weight = dVCM / (dVCM + camera_pdf_factor * dVC)
/// where camera_pdf_factor accounts for the camera's directional PDF.
fn camera_connection(
    hit_pos: vec3<f32>,
    hit_normal: vec3<f32>,
    wo: vec3<f32>,
    hit_tangent: vec4<f32>,
    throughput: vec3<f32>,
    material: ResolvedRayHitFull,
    dVCM: f32,
    dVC: f32,
    dVM: f32,
) {
    // Direction from the hit point to the camera
    let to_camera = view.world_position - hit_pos;
    let dist_sq = dot(to_camera, to_camera);
    if dist_sq < 1e-10 {
        return;
    }
    let dist = sqrt(dist_sq);
    let dir_to_camera = to_camera / dist;

    // Check that the surface faces the camera
    let cos_at_surface = dot(hit_normal, dir_to_camera);
    if cos_at_surface <= 0.0 {
        return;
    }

    // Project the hit point onto the image plane
    let clip = view.clip_from_world * vec4(hit_pos, 1.0);

    // Behind camera check
    if clip.w <= 0.0 {
        return;
    }

    let ndc = clip.xy / clip.w;

    // NDC is in [-1, 1], convert to pixel coordinates
    // Note: Bevy uses Y-down in NDC for the pathtracer (pixel_ndc.y is negated
    // in the camera ray generation), so we negate Y here too.
    let pixel_f = (vec2(ndc.x, -ndc.y) * 0.5 + 0.5) * vec2(f32(vcm_params.screen_width), f32(vcm_params.screen_height));

    // Bounds check (with 0.5 pixel margin to avoid edge artifacts)
    if pixel_f.x < 0.0 || pixel_f.x >= f32(vcm_params.screen_width)
        || pixel_f.y < 0.0 || pixel_f.y >= f32(vcm_params.screen_height) {
        return;
    }

    let pixel = vec2<u32>(u32(pixel_f.x), u32(pixel_f.y));

    // Visibility test: shadow ray from hit point to camera
    // We construct a point slightly in front of the camera to use with
    // trace_point_visibility (which expects two surface points).
    let camera_point = view.world_position;
    let visibility = trace_point_visibility(hit_pos, camera_point);
    if visibility <= 0.0 {
        return;
    }

    // Evaluate BRDF at the light vertex toward the camera direction.
    // wo = direction light arrived from (toward previous vertex)
    // dir_to_camera = direction toward camera
    let brdf = evaluate_brdf(hit_normal, wo, dir_to_camera, material.material);

    // Camera PDF computation for pinhole camera.
    // For a pinhole camera, the directional PDF in solid angle is:
    //   pdf_camera_dir = 1 / (pixel_solid_angle)
    // where pixel_solid_angle = pixel_area_on_sensor / focal_length^2
    //
    // The image plane factor relates the camera direction to the sensor:
    //   cos_at_camera = dot(camera_forward, dir_to_camera) (note: negative since
    //   camera looks along -Z in view space)
    //
    // We compute cos_at_camera from the view matrix. The camera forward
    // direction in world space is the third row of view_from_world (negated
    // because the camera looks along -Z in view space).
    let camera_forward = -normalize(view.world_from_view[2].xyz);
    let cos_at_camera = dot(camera_forward, -dir_to_camera);

    if cos_at_camera <= 0.0 {
        return;
    }

    // For a pinhole camera with the projection matrix available:
    // The image plane is at distance d = 1/tan(fov/2) in NDC-normalized
    // units. The pixel area in solid angle at the camera is:
    //   pixel_solid_angle = 1 / (W * H * f_x * f_y * cos^3(theta))
    // where f_x = clip_from_view[0][0], f_y = clip_from_view[1][1]
    // represent the focal lengths in NDC.
    let f_x = abs(view.clip_from_view[0][0]);
    let f_y = abs(view.clip_from_view[1][1]);
    let screen_w = f32(vcm_params.screen_width);
    let screen_h = f32(vcm_params.screen_height);

    // The "image-to-solid-angle" factor (inverse of pixel solid angle):
    //   pdf_camera_W = f_x * f_y * W * H / (4 * cos^3(theta))
    // The factor of 4 accounts for the [-1,1]^2 NDC range mapping to [0,W]*[0,H].
    // But with our projection convention (Bevy), the relationship is:
    //   pdf_camera_W = (f_x * screen_w / 2) * (f_y * screen_h / 2) / cos^3
    let cos3 = cos_at_camera * cos_at_camera * cos_at_camera;
    let pdf_camera_W = f_x * f_y * screen_w * screen_h / (4.0 * cos3);

    // Convert from solid angle PDF to area PDF at the hit point:
    //   pdf_camera_A = pdf_camera_W * cos_at_surface / dist^2
    let pdf_camera_A = pdf_camera_W * cos_at_surface / dist_sq;

    // MIS weight for camera connection (t=1 strategy).
    // Following SmallVCM: the camera has only one vertex (the eye), so
    // the MIS weight accounts for alternative strategies that could have
    // generated the same path:
    //   w_light = pdf_camera_A * (dVCM + dVC * pdf_camera_W_for_connection)
    // But for pinhole cameras, the connection from camera to this vertex
    // has no reverse PDF (pinhole is a delta), so the weight simplifies.
    //
    // SmallVCM uses:
    //   mis_weight = 1 / (dVCM * pdf_camera_A + dVC * pdf_camera_A_times_reverse + 1)
    // For pinhole (no camera path continuation possible from t=1):
    //   w_light = dVCM * pdf_camera_A + dVC * pdf_camera_A (simplified)
    // Actually from SmallVCM ConnectToCamera:
    //   cameraPdfA = ImageToSolidAngle / dist^2 * cos_at_surface
    //   wLight = cameraPdfA * (dVCM + dVC * vcWeightFactor)
    //   misWeight = 1 / (wLight + 1)
    //
    // vcWeightFactor here is vcm_params.vc_weight which is 1/vm_weight.
    // But SmallVCM's ConnectToCamera actually doesn't use vc_weight for this.
    // Looking more carefully: in SmallVCM, the camera has no reverse PDF
    // for continuation, so:
    //   wLight = cameraPdfA * dVCM + cameraPdfA * dVC * 0 (no camera continuation)
    //          = cameraPdfA * dVCM
    // Wait, SmallVCM does: wLight = aCameraPdfA * (aLightVertex.dVCM + aLightVertex.dVC * mLightSubPathCount)
    // where mLightSubPathCount is num_light_paths. This is because the pinhole
    // camera's "reverse PDF" when projected back is effectively the image plane
    // PDF times num_light_paths due to the MIS normalization convention.
    //
    // Actually from SmallVCM source (ConnectToCamera):
    //   wLight = cameraPdfA / lightPathCount * (dVCM + dVC * cameraPdfA_reverse)
    //
    // Let me use the simplified form that works with balance heuristic:
    let w_light = sanitize_mis(pdf_camera_A * (dVCM + dVC * f32(vcm_params.num_light_paths)));
    let mis_weight = clamp(1.0 / (w_light + 1.0), 0.0, 1.0);

    // Geometry term: cos_at_surface / dist^2
    // Note: evaluate_brdf already includes cos_at_surface (it multiplies by
    // saturate(dot(N, L))), so the explicit geometry term is just 1/dist^2.
    // Note: do NOT apply view.exposure here — the camera trace shader
    // applies exposure uniformly to all radiance (including splatted
    // contributions) after compositing.
    let contribution = mis_weight * throughput * brdf / (dist_sq * f32(vcm_params.num_light_paths));

    if is_valid_radiance(contribution) {
        splat_add(pixel, contribution);
    }
}

/// Importance-samples a new bounce direction from the material BRDF at the
/// given hit point. This is the same logic as in pathtracer.wgsl but
/// extracted here so light tracing can reuse it.
///
/// Returns: NextBounce with sampled direction, forward PDF, and specular flag.
struct LightTraceBounce {
    /// Sampled outgoing direction (direction of continued light travel).
    wi: vec3<f32>,
    /// Forward PDF p(wi | wo) under the mixed diffuse/specular strategy.
    pdf: f32,
    /// True when the surface is a perfect mirror.
    perfectly_specular_bounce: bool,
}

fn importance_sample_bounce(wo: vec3<f32>, ray_hit: ResolvedRayHitFull, rng: ptr<function, u32>) -> LightTraceBounce {
    let is_perfectly_specular = ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
    if is_perfectly_specular {
        return LightTraceBounce(reflect(-wo, ray_hit.world_normal), 1.0, true);
    }

    let diffuse_weight = mix(mix(0.4, 0.9, ray_hit.material.perceptual_roughness), 0.0, ray_hit.material.metallic);
    let specular_weight = 1.0 - diffuse_weight;

    let TBN = calculate_tbn_mikktspace(ray_hit.world_normal, ray_hit.world_tangent);
    let T = TBN[0];
    let B = TBN[1];
    let N = TBN[2];

    let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));

    var wi: vec3<f32>;
    var wi_tangent: vec3<f32>;
    let diffuse_selected = rand_f(rng) < diffuse_weight;
    if diffuse_selected {
        wi = sample_cosine_hemisphere(ray_hit.world_normal, rng);
        wi_tangent = vec3(dot(wi, T), dot(wi, B), dot(wi, N));
    } else {
        wi_tangent = sample_ggx_vndf(wo_tangent, ray_hit.material.roughness, rng);
        wi = wi_tangent.x * T + wi_tangent.y * B + wi_tangent.z * N;
    }

    let diffuse_pdf = max(0.0, dot(wi, ray_hit.world_normal)) / PI;
    let specular_pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, ray_hit.material.roughness);
    let pdf = (diffuse_weight * diffuse_pdf) + (specular_weight * specular_pdf);

    return LightTraceBounce(wi, pdf, false);
}

/// Main light tracing entry point — one invocation per light subpath.
///
/// Each thread:
/// 1. Samples a light source and emission direction
/// 2. Traces the light subpath through the scene
/// 3. At each non-specular hit, stores a LightPathVertex for later
///    use in vertex connections and merging
/// 4. Updates VCM MIS partial weights (dVCM, dVC, dVM) at each bounce
@compute @workgroup_size(64, 1, 1)
fn light_trace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    if thread_id >= vcm_params.num_light_paths {
        return;
    }

    // Initialize RNG with a seed distinct from camera paths.
    // Use a large prime offset so light and camera paths are uncorrelated.
    var rng = thread_id + vcm_params.iteration * 7919u + 2654435761u;

    // Sample emission from a light source
    let emission = sample_light_emission(&rng);

    // Skip directional lights for now (Phase 1 — will be handled later)
    if emission.is_directional {
        return;
    }

    // Guard against degenerate emission
    if emission.pdf_position <= 0.0 || emission.pdf_direction <= 0.0 || emission.pdf_light_pick <= 0.0 {
        return;
    }

    // Initialize throughput from emission
    let emission_pdf_combined = emission.pdf_position * emission.pdf_direction * emission.pdf_light_pick;
    var throughput = emission.radiance / emission_pdf_combined;
    if !is_valid(throughput) { return; }

    // Initialize VCM MIS partial weights from emission PDFs.
    // Following SmallVCM: at the first vertex after emission,
    //   dVCM = pdf_position / (pdf_direction * pdf_light_pick)
    // This accounts for the probability of having generated this
    // particular light subpath start.
    var dVCM = emission.pdf_position / (emission.pdf_direction * emission.pdf_light_pick);
    var dVC = 0.0;
    var dVM = 0.0;

    // Trace from the emission point in the sampled direction
    var ray_origin = emission.position;
    var ray_direction = emission.direction;
    var ray_t_min = RAY_T_MIN;

    for (var bounce = 0u; bounce < MAX_LIGHT_BOUNCES; bounce += 1u) {
        let ray = trace_ray(ray_origin, ray_direction, ray_t_min, RAY_T_MAX, RAY_FLAG_NONE);

        // Miss — light escaped the scene
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            break;
        }

        let ray_hit = resolve_ray_hit_full(ray);

        // wo = direction pointing back toward the previous vertex (opposite of
        // the ray direction we traveled). This is the "outgoing" direction when
        // we think of light arriving at this vertex.
        let wo = -ray_direction;

        // Geometry term components for MIS weight conversion at vertex arrival.
        // SmallVCM converts the partial weights from solid-angle measure at
        // the previous vertex to area measure at the current vertex:
        //   dVCM *= dist^2 / cos_theta_hit
        //   dVC  /= cos_theta_hit
        //   dVM  /= cos_theta_hit
        // Using balance heuristic where MIS(x) = x.
        let hit_distance = ray.t;
        let cos_theta_hit = abs(dot(ray_hit.world_normal, wo));

        if cos_theta_hit > 1e-6 {
            dVCM *= (hit_distance * hit_distance) / cos_theta_hit;
            dVC /= cos_theta_hit;
            dVM /= cos_theta_hit;
        } else {
            // Grazing angle -- MIS weights become degenerate
            dVCM = 0.0;
            dVC = 0.0;
            dVM = 0.0;
        }

        // Sanitize MIS weights after geometry term conversion
        dVCM = sanitize_mis(dVCM);
        dVC = sanitize_mis(dVC);
        dVM = sanitize_mis(dVM);

        let specular = is_specular_surface(ray_hit.material.roughness, ray_hit.material.metallic);

        // Store vertex for non-specular surfaces
        if !specular {
            let vertex_index = atomicAdd(&light_vertex_count, 1u);
            if vertex_index < vcm_params.max_light_vertices {
                var vertex: LightPathVertex;
                vertex.position_and_perceptual_roughness = vec4(ray_hit.world_position, ray_hit.material.perceptual_roughness);
                vertex.normal_and_metallic = vec4(ray_hit.world_normal, ray_hit.material.metallic);
                vertex.throughput_and_roughness = vec4(throughput, ray_hit.material.roughness);
                vertex.base_color_and_reflectance = vec4(
                    ray_hit.material.base_color,
                    (ray_hit.material.reflectance.x + ray_hit.material.reflectance.y + ray_hit.material.reflectance.z) / 3.0
                );
                vertex.incoming_direction_and_dVCM = vec4(wo, sanitize_mis(dVCM));
                vertex.world_tangent = ray_hit.world_tangent;
                vertex.dVC = sanitize_mis(dVC);
                vertex.dVM = sanitize_mis(dVM);
                light_vertices[vertex_index] = vertex;
            }

            // Camera connection: connect this light vertex directly to
            // the camera (t=1 strategy) and splat the contribution.
            camera_connection(
                ray_hit.world_position,
                ray_hit.world_normal,
                wo,
                ray_hit.world_tangent,
                throughput,
                ray_hit,
                dVCM, dVC, dVM,
            );
        }

        // Sample next bounce direction
        let next_bounce = importance_sample_bounce(wo, ray_hit, &rng);

        if next_bounce.pdf <= 0.0 {
            break;
        }

        // For specular bounces, zero out dVC and dVM (delta BSDF)
        if next_bounce.perfectly_specular_bounce {
            // On a specular surface, the path is deterministic — no connections
            // or merging can happen here. MIS weights for those techniques = 0.
            dVC = 0.0;
            dVM = 0.0;
            // dVCM stays as-is (still needed for subsequent non-specular vertices)
        } else {
            // Compute forward and reverse PDFs for MIS weight update
            let TBN = calculate_tbn_mikktspace(ray_hit.world_normal, ray_hit.world_tangent);
            let T = TBN[0];
            let B = TBN[1];
            let N = TBN[2];
            let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
            let wi_tangent = vec3(dot(next_bounce.wi, T), dot(next_bounce.wi, B), dot(next_bounce.wi, N));

            let brdf_eval = evaluate_brdf_bidirectional(
                ray_hit.world_normal, wo, next_bounce.wi,
                wo_tangent, wi_tangent,
                ray_hit.material
            );

            let fwd_pdf = brdf_eval.pdf_forward;
            let rev_pdf = brdf_eval.pdf_reverse;

            if fwd_pdf <= 0.0 {
                break;
            }

            // cos(theta) of the sampled direction with the surface normal
            let cos_theta = abs(dot(ray_hit.world_normal, next_bounce.wi));

            // Update MIS partial weights using VCM recursive formulas.
            // These formulas come from SmallVCM / Georgiev et al. 2012.
            //
            // The key idea: each partial weight tracks the ratio of
            // probabilities of alternative sampling strategies to the
            // probability of the current strategy.
            //
            // new_dVCM = 1 / fwd_pdf
            //   (probability ratio for the "next vertex" being sampled
            //    differently)
            //
            // new_dVC = (cos_theta / fwd_pdf) * (dVCM + dVC * rev_pdf + vm_weight)
            //   (accumulates the vertex connection weight through the path)
            //
            // new_dVM = (cos_theta / fwd_pdf) * (dVCM * vc_weight + dVM * rev_pdf + 1.0)
            //   (accumulates the vertex merging weight through the path)
            let factor = cos_theta / fwd_pdf;
            let new_dVCM = 1.0 / fwd_pdf;
            let new_dVC = factor * (dVCM + dVC * rev_pdf + vcm_params.vm_weight);
            let new_dVM = factor * (dVCM * vcm_params.vc_weight + dVM * rev_pdf + 1.0);

            dVCM = sanitize_mis(new_dVCM);
            dVC = sanitize_mis(new_dVC);
            dVM = sanitize_mis(new_dVM);
        }

        // Update throughput.
        // Note: evaluate_brdf already includes cos(theta) in its result
        // (both diffuse and specular lobes multiply by saturate(dot(N, L))),
        // so we do NOT multiply by cos_theta again here.
        let brdf = evaluate_brdf(ray_hit.world_normal, wo, next_bounce.wi, ray_hit.material);
        throughput *= brdf / next_bounce.pdf;
        if !is_valid(throughput) { break; }

        // Russian roulette for early termination (after at least 3 bounces)
        if bounce >= 2u {
            let p = min(1.0, luminance(throughput));
            if rand_f(&rng) > p { break; }
            throughput /= p;
        }

        // Set up next ray
        ray_origin = ray_hit.world_position;
        ray_direction = next_bounce.wi;
        ray_t_min = RAY_T_MIN;
    }
}
