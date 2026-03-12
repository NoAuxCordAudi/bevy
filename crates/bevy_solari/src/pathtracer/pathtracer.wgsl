enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::pbr_functions::calculate_tbn_mikktspace
#import bevy_pbr::utils::{rand_f, rand_vec2f, sample_cosine_hemisphere, halton_2d, sample_disk}
#import bevy_render::maths::PI
#import bevy_render::view::View
#import bevy_solari::brdf::{evaluate_brdf, fresnel_dielectric, refract_ray, is_total_internal_reflection}
#import bevy_solari::sampling::{sample_random_light, random_emissive_light_pdf, sample_ggx_vndf, ggx_vndf_pdf, power_heuristic}
#import bevy_solari::scene_bindings::{trace_ray, resolve_ray_hit_full, ResolvedRayHitFull, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD, directional_lights, light_sources}
#import bevy_solari::sky::{evaluate_sky_ambient, evaluate_sun_disk}

/// Checks whether a vec3 contains only finite, non-NaN values within a
/// reasonable magnitude. Used as a safety check to discard degenerate
/// radiance or throughput values that could corrupt the accumulation buffer.
fn is_valid(v: vec3<f32>) -> bool {
    return all(v == clamp(v, vec3(-1e20), vec3(1e20)));
}

const MAX_BOUNCES = 128u;

struct PathtracerSettings {
    min_samples: u32,
    max_samples: u32,
    convergence_threshold: f32,
    aperture_radius: f32,
    focal_distance: f32,
}

@group(1) @binding(0) var accumulation_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> view: View;
@group(1) @binding(3) var variance_texture: texture_storage_2d<r32float, read_write>;
@group(1) @binding(4) var<uniform> settings: PathtracerSettings;

/// Main compute shader entry point – one invocation per pixel.
///
/// Each invocation traces a single camera ray through the scene and
/// progressively accumulates the result into `accumulation_texture` using a
/// running average. The high-level algorithm:
///
///  1. Load the previous accumulated color and sample count from the
///     accumulation texture.
///  2. Generate a jittered sub-pixel camera ray for anti-aliasing.
///  3. Perform a path-tracing loop (up to `MAX_BOUNCES`):
///     a. Cast a ray against the scene TLAS.
///     b. On hit, evaluate emissive contribution with MIS weighting.
///     c. Sample direct lighting (next-event estimation) with MIS for
///        non-mirror surfaces.
///     d. Importance-sample the BRDF to pick the next bounce direction.
///     e. Apply Russian roulette for unbiased early termination.
///  4. Apply camera exposure and blend the new sample into the running
///     average.
///  5. Write the updated accumulation value and the current result to the
///     view output.
@compute @workgroup_size(8, 8, 1)
fn pathtrace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    let old_color = textureLoad(accumulation_texture, global_id.xy);

    // Early-out for converged / max-sample pixels
    let sample_count = u32(old_color.a);
    if sample_count >= settings.max_samples {
        textureStore(view_output, global_id.xy, vec4(old_color.rgb, 1.0));
        return;
    }

    // Setup RNG
    let pixel_index = global_id.x + global_id.y * u32(view.viewport.z);
    let frame_index = u32(old_color.a) * 5782582u;
    var rng = pixel_index + frame_index;

    // Shoot the first ray from the camera
    let pixel_center = vec2<f32>(global_id.xy) + 0.5;
    let jitter = halton_2d(u32(old_color.a)) - 0.5;
    let pixel_uv = (pixel_center + jitter) / view.viewport.zw;
    let pixel_ndc = (pixel_uv * 2.0) - 1.0;
    let primary_ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
    var ray_origin = view.world_position;
    var ray_direction = normalize((primary_ray_target.xyz / primary_ray_target.w) - ray_origin);

    // Depth of field: thin lens approximation
    if settings.aperture_radius > 0.0 {
        let focal_point = ray_origin + settings.focal_distance * ray_direction;
        let camera_right = view.world_from_view[0].xyz;
        let camera_up = view.world_from_view[1].xyz;
        let disk_offset = sample_disk(settings.aperture_radius, &rng);
        ray_origin = ray_origin + disk_offset.x * camera_right + disk_offset.y * camera_up;
        ray_direction = normalize(focal_point - ray_origin);
    }

    var ray_t_min = 0.0;

    // Path trace
    var radiance = vec3(0.0);
    var throughput = vec3(1.0);
    var p_bounce = 0.0;
    var bounce_was_perfect_reflection = true;
    var bounce_count = 0u;
    loop {
        if bounce_count >= MAX_BOUNCES { break; }
        bounce_count += 1u;
        let ray = trace_ray(ray_origin, ray_direction, ray_t_min, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind != RAY_QUERY_INTERSECTION_NONE {
            let ray_hit = resolve_ray_hit_full(ray);
            let wo = -ray_direction;

            var mis_weight = 1.0;
            if !bounce_was_perfect_reflection {
                let p_light = random_emissive_light_pdf(ray_hit);
                mis_weight = power_heuristic(p_bounce, p_light);
            }
            radiance += mis_weight * throughput * ray_hit.material.emissive;

            // Sample direct lighting, but only if the surface is not mirror-like or dielectric
            let is_dielectric = ray_hit.material.specular_transmission > 0.5;
            let is_perfectly_specular = (ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999) || is_dielectric;
            if !is_perfectly_specular {
                let direct_lighting = sample_random_light(ray_hit.world_position, ray_hit.world_normal, &rng);

                mis_weight = 1.0;
                if direct_lighting.brdf_rays_can_hit {
                    let pdf_of_bounce = brdf_pdf(wo, direct_lighting.wi, ray_hit);
                    mis_weight = power_heuristic(1.0 / direct_lighting.inverse_pdf, pdf_of_bounce);
                }

                let direct_lighting_brdf = evaluate_brdf(ray_hit.world_normal, wo, direct_lighting.wi, ray_hit.material);
                radiance += mis_weight * throughput * direct_lighting.radiance * direct_lighting.inverse_pdf * direct_lighting_brdf;
            }

            // Sample new ray direction from the material BRDF for next bounce
            let next_bounce = importance_sample_next_bounce(wo, ray_hit, &rng);
            ray_direction = next_bounce.wi;
            p_bounce = next_bounce.pdf;
            bounce_was_perfect_reflection = next_bounce.perfectly_specular_bounce;

            if next_bounce.is_dielectric {
                // Dielectric: offset origin to correct side of surface and tint by base color
                let normal_sign = sign(dot(next_bounce.wi, ray_hit.geometric_world_normal));
                ray_origin = ray_hit.world_position + normal_sign * ray_hit.geometric_world_normal * RAY_T_MIN * 2.0;
                ray_t_min = RAY_T_MIN;
                throughput *= ray_hit.material.base_color;
            } else {
                // Standard BRDF bounce
                ray_origin = ray_hit.world_position;
                ray_t_min = RAY_T_MIN;
                let brdf = evaluate_brdf(ray_hit.world_normal, wo, next_bounce.wi, ray_hit.material);
                throughput *= brdf / next_bounce.pdf;
            }
            if !is_valid(throughput) { break; }

            // Russian roulette for early termination
            let p = min(1.0, luminance(throughput));
            if rand_f(&rng) > p { break; }
            throughput /= p;
        } else {
            // Sky environment contribution for escaped rays
            radiance += throughput * evaluate_sky_ambient(ray_direction);

            // Sun disk with MIS weighting against direct light sampling
            let sun_radiance = evaluate_sun_disk(ray_direction);
            if any(sun_radiance > vec3(0.0)) {
                var mis_weight = 1.0;
                if !bounce_was_perfect_reflection {
                    let total_light_count = arrayLength(&light_sources);
                    let sun = directional_lights[0];
                    let p_light = 1.0 / (sun.inverse_pdf * f32(total_light_count));
                    mis_weight = power_heuristic(p_bounce, p_light);
                }
                radiance += throughput * mis_weight * sun_radiance;
            }
            break;
        }
    }

    if !is_valid(radiance) { radiance = vec3(0.0); }

    // Camera exposure
    radiance *= view.exposure;

    // Accumulation over time via running average
    let new_color = mix(old_color.rgb, radiance, 1.0 / (old_color.a + 1.0));
    let new_sample_count = sample_count + 1u;

    // Welford's online variance update (on luminance)
    let lum = luminance(radiance);
    let old_mean_lum = luminance(old_color.rgb);
    let new_mean_lum = luminance(new_color);
    let old_m2 = textureLoad(variance_texture, global_id.xy).r;
    let new_m2 = old_m2 + (lum - old_mean_lum) * (lum - new_mean_lum);

    // Convergence: standard error of the mean with absolute floor for dark pixels
    var final_sample_count = new_sample_count;
    let n = f32(new_sample_count);
    if new_sample_count >= settings.min_samples && settings.convergence_threshold > 0.0 {
        let variance = max(new_m2 / (n - 1.0), 0.0);
        let standard_error = sqrt(variance / n);
        let threshold = settings.convergence_threshold;
        // Bright pixels: relative standard error (SE < threshold * mean)
        // Dark pixels: absolute standard error (SE < threshold * absolute_floor)
        // The absolute_floor prevents premature convergence of near-black pixels
        let absolute_floor = 0.01;
        let converged = standard_error < threshold * max(new_mean_lum, absolute_floor);
        if converged {
            final_sample_count = settings.max_samples;
        }
    }

    textureStore(variance_texture, global_id.xy, vec4(new_m2, 0.0, 0.0, 0.0));
    textureStore(accumulation_texture, global_id.xy, vec4(new_color, f32(final_sample_count)));
    textureStore(view_output, global_id.xy, vec4(new_color, 1.0));
}

/// Result of importance-sampling the BRDF at a surface hit point.
struct NextBounce {
    /// The sampled outgoing light direction (toward the next bounce).
    wi: vec3<f32>,
    /// Probability density of having sampled `wi` under the mixed
    /// diffuse/specular sampling strategy. Set to 1.0 for perfect mirrors
    /// and dielectrics.
    pdf: f32,
    /// `true` when the surface is a perfect mirror or dielectric, meaning
    /// no direct-lighting MIS is needed for emissive hits on the next bounce.
    perfectly_specular_bounce: bool,
    /// `true` when the bounce is through a dielectric (glass) surface.
    is_dielectric: bool,
}

/// Importance-samples a new bounce direction from the material BRDF at the
/// given hit point.
///
/// For perfect mirrors (roughness ≤ `MIRROR_ROUGHNESS_THRESHOLD` and fully
/// metallic) the function returns a deterministic mirror reflection with
/// `pdf = 1.0`.
///
/// For all other materials a mixed sampling strategy is used:
///  - With probability `diffuse_weight`, a cosine-weighted hemisphere sample
///    is drawn (good for diffuse lobes).
///  - Otherwise, a GGX VNDF (visible normal distribution function) sample is
///    drawn (good for specular lobes).
///
/// The combined PDF is the weighted sum of both strategies so it can be used
/// for MIS with direct-light sampling.
fn importance_sample_next_bounce(wo: vec3<f32>, ray_hit: ResolvedRayHitFull, rng: ptr<function, u32>) -> NextBounce {
    // Dielectric (glass) materials: refract or reflect based on Fresnel
    let is_dielectric = ray_hit.material.specular_transmission > 0.5;
    if is_dielectric {
        let front_face = dot(wo, ray_hit.geometric_world_normal) > 0.0;
        let eta = select(ray_hit.material.ior, 1.0 / ray_hit.material.ior, front_face);
        let n = select(-ray_hit.geometric_world_normal, ray_hit.geometric_world_normal, front_face);

        let cos_theta_i = min(dot(wo, n), 1.0);
        let total_internal_reflection = is_total_internal_reflection(cos_theta_i, eta);
        let fresnel = fresnel_dielectric(cos_theta_i, eta);

        var wi: vec3<f32>;
        if total_internal_reflection || rand_f(rng) < fresnel {
            wi = reflect(-wo, n);
        } else {
            wi = refract_ray(-wo, n, eta);
        }
        return NextBounce(wi, 1.0, true, true);
    }

    let is_perfectly_specular = ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
    if is_perfectly_specular {
        return NextBounce(reflect(-wo, ray_hit.world_normal), 1.0, true, false);
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

    return NextBounce(wi, pdf, false, false);
}

/// Evaluates the probability density of sampling direction `wi` given
/// outgoing direction `wo` at the surface described by `ray_hit`, under the
/// same mixed diffuse/specular strategy used by
/// `importance_sample_next_bounce`.
///
/// This is needed for multiple importance sampling (MIS): when a light
/// sample direction is chosen by direct-light sampling, we need to know what
/// PDF the BRDF sampling strategy would have assigned to that same direction
/// in order to compute the MIS weight via the power heuristic.
fn brdf_pdf(wo: vec3<f32>, wi: vec3<f32>, ray_hit: ResolvedRayHitFull) -> f32 {
    if ray_hit.material.specular_transmission > 0.5 {
        return 0.0;
    }
    let diffuse_weight = mix(mix(0.4, 0.9, ray_hit.material.perceptual_roughness), 0.0, ray_hit.material.metallic);
    let specular_weight = 1.0 - diffuse_weight;

    let TBN = calculate_tbn_mikktspace(ray_hit.world_normal, ray_hit.world_tangent);
    let T = TBN[0];
    let B = TBN[1];
    let N = TBN[2];

    let wo_tangent = vec3(dot(wo, T), dot(wo, B), dot(wo, N));
    let wi_tangent = vec3(dot(wi, T), dot(wi, B), dot(wi, N));

    let diffuse_pdf = max(0.0, wi_tangent.z) / PI;
    let specular_pdf = ggx_vndf_pdf(wo_tangent, wi_tangent, ray_hit.material.roughness);
    let pdf = (diffuse_weight * diffuse_pdf) + (specular_weight * specular_pdf);
    return pdf;
}
