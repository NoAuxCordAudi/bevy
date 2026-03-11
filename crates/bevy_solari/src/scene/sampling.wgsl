enable wgpu_ray_query;

#define_import_path bevy_solari::sampling

#import bevy_pbr::lighting::D_GGX
#import bevy_pbr::pbr_functions::calculate_tbn_mikktspace
#import bevy_pbr::utils::{rand_f, rand_vec2f, rand_u, rand_range_u, sample_cosine_hemisphere}
#import bevy_render::maths::{PI, PI_2, orthonormalize}
#import bevy_solari::scene_bindings::{trace_ray, RAY_T_MIN, RAY_T_MAX, light_sources, directional_lights, LightSource, LIGHT_SOURCE_KIND_DIRECTIONAL, LIGHT_SOURCE_KIND_EMISSIVE_MESH, resolve_triangle_data_full, ResolvedRayHitFull, MIRROR_ROUGHNESS_THRESHOLD}

fn power_heuristic(f: f32, g: f32) -> f32 {
    return balance_heuristic(f * f, g * g);
}

fn balance_heuristic(f: f32, g: f32) -> f32 {
    let sum = f + g;
    if sum == 0.0 {
        return 0.0;
    }
    return max(0.0, f / sum);
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 1)
fn sample_ggx_vndf(wi_tangent: vec3<f32>, roughness: f32, rng: ptr<function, u32>) -> vec3<f32> {
    // Mirror BRDF case
    if roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        return vec3(-wi_tangent.xy, wi_tangent.z);
    }

    let i = wi_tangent;
    let rand = rand_vec2f(rng);
    let i_std = normalize(vec3(i.xy * roughness, i.z));
    let phi = PI_2 * rand.x;
    let a = roughness;
    let s = 1.0 + length(vec2(i.xy));
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    let b = select(i_std.z, k * i_std.z, i.z > 0.0);
    let z = fma(1.0 - rand.y, 1.0 + b, -b);
    let sin_theta = sqrt(saturate(1.0 - z * z));
    let o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    let m_std = i_std + o_std;
    let m = normalize(vec3(m_std.xy * roughness, m_std.z));
    return 2.0 * dot(i, m) * m - i;
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 2)
fn ggx_vndf_pdf(wi_tangent: vec3<f32>, wo_tangent: vec3<f32>, roughness: f32) -> f32 {
    // Mirror BRDF case
    if roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        let mirror_wo = vec3(-wi_tangent.xy, wi_tangent.z);
        return f32(all(abs(mirror_wo - wo_tangent) < vec3(0.0001)));
    }

    let i = wi_tangent;
    let o = wo_tangent;
    let m = normalize(i + o);
    let ndf = D_GGX(roughness, saturate(m.z));
    let ai = roughness * i.xy;
    let len2 = dot(ai, ai);
    let t = sqrt(len2 + i.z * i.z);
    if i.z >= 0.0 {
        let a = roughness;
        let s = 1.0 + length(i.xy);
        let a2 = a * a;
        let s2 = s * s;
        let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        return ndf / (2.0 * (k * i.z + t));
    }
    return ndf * (t - i.z) / (2.0 * len2);
}

const NULL_LIGHT_ID = 0xFFFFFFFFu;

struct LightSample {
    light_id: u32,
    seed: u32,
}

struct ResolvedLightSample {
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    radiance: vec3<f32>,
    inverse_pdf: f32,
}

struct LightContribution {
    radiance: vec3<f32>,
    inverse_pdf: f32,
    wi: vec3<f32>,
    brdf_rays_can_hit: bool,
}

struct LightContributionNoPdf {
    radiance: vec3<f32>,
    wi: vec3<f32>,
}

struct GenerateRandomLightSampleResult {
    light_sample: LightSample,
    resolved_light_sample: ResolvedLightSample,
}

fn sample_random_light(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_random_light_sample(rng);
    var light_contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    light_contribution.radiance *= trace_light_visibility(ray_origin, sample.resolved_light_sample.world_position);
    return light_contribution;
}

fn random_emissive_light_pdf(hit: ResolvedRayHitFull) -> f32 {
    let light_count = arrayLength(&light_sources);
    return 1.0 / (f32(light_count) * f32(hit.triangle_count) * hit.triangle_area);
}

fn generate_random_light_sample(rng: ptr<function, u32>) -> GenerateRandomLightSampleResult {
    let light_count = arrayLength(&light_sources);
    let light_id = rand_range_u(light_count, rng);

    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if light_source.kind != LIGHT_SOURCE_KIND_DIRECTIONAL {
        let triangle_count = light_source.kind >> 1u;
        triangle_id = rand_range_u(triangle_count, rng);
    }

    let seed = rand_u(rng);
    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= f32(light_count);

    return GenerateRandomLightSampleResult(light_sample, resolved_light_sample);
}

fn resolve_light_sample(light_sample: LightSample, light_source: LightSource) -> ResolvedLightSample {
    if light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
        let directional_light = directional_lights[light_source.id];

#ifndef NO_DIRECTIONAL_LIGHT_SOFT_SHADOWS
        // Sample a random direction within a cone whose base is the sun approximated as a disk
        // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec30%3A305
        var rng = light_sample.seed;
        let random = rand_vec2f(&rng);
        let cos_theta = (1.0 - random.x) + random.x * directional_light.cos_theta_max;
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        let phi = random.y * PI_2;
        let x = cos(phi) * sin_theta;
        let y = sin(phi) * sin_theta;
        var direction_to_light = vec3(x, y, cos_theta);

        // Rotate the ray so that the cone it was sampled from is aligned with the light direction
        direction_to_light = orthonormalize(directional_light.direction_to_light) * direction_to_light;
#else
        let direction_to_light = directional_light.direction_to_light;
#endif

        return ResolvedLightSample(
            vec4(direction_to_light, 0.0),
            -direction_to_light,
            directional_light.luminance,
            directional_light.inverse_pdf,
        );
    } else {
        let triangle_count = light_source.kind >> 1u;
        let triangle_id = light_sample.light_id & 0xFFFFu;
        let barycentrics = triangle_barycentrics(light_sample.seed);
        let triangle_data = resolve_triangle_data_full(light_source.id, triangle_id, barycentrics);

        return ResolvedLightSample(
            vec4(triangle_data.world_position, 1.0),
            triangle_data.world_normal,
            triangle_data.material.emissive.rgb,
            f32(triangle_count) * triangle_data.triangle_area,
        );
    }
}

fn calculate_resolved_light_contribution(resolved_light_sample: ResolvedLightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContribution {
    let ray = resolved_light_sample.world_position.xyz - (resolved_light_sample.world_position.w * ray_origin);
    let light_distance = length(ray);
    let wi = ray / light_distance;

    let cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
    let light_distance_squared = light_distance * light_distance;

    let radiance = resolved_light_sample.radiance * (cos_theta_light / light_distance_squared);

    return LightContribution(radiance, resolved_light_sample.inverse_pdf, wi, resolved_light_sample.world_position.w == 1.0);
}

fn resolve_and_calculate_light_contribution(light_sample: LightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContributionNoPdf {
    let resolved_light_sample = resolve_light_sample(light_sample, light_sources[light_sample.light_id >> 16u]);
    let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, ray_origin, origin_world_normal);
    return LightContributionNoPdf(light_contribution.radiance, light_contribution.wi);
}

fn trace_light_visibility(ray_origin: vec3<f32>, light_sample_world_position: vec4<f32>) -> f32 {
    var ray_direction = light_sample_world_position.xyz;
    var ray_t_max = RAY_T_MAX;

    if light_sample_world_position.w == 1.0 {
        let ray = ray_direction - ray_origin;
        let dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN - RAY_T_MIN;
    }

    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn trace_point_visibility(ray_origin: vec3<f32>, point: vec3<f32>) -> f32 {
    let ray = point - ray_origin;
    let dist = length(ray);
    let ray_direction = ray / dist;

    let ray_t_max = dist - RAY_T_MIN - RAY_T_MIN;
    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
fn triangle_barycentrics(seed: u32) -> vec3<f32> {
    var rng = seed;
    var barycentrics = rand_vec2f(&rng);
    if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}

// ---------------------------------------------------------------------------
// Reverse PDF infrastructure (Phase 6.3)
// ---------------------------------------------------------------------------

/// Computes the reverse PDF p(wi | wo) under the mixed diffuse/specular
/// sampling strategy, i.e., the probability that the BRDF importance sampler
/// would have generated `wi` if `wo` were treated as the outgoing direction.
///
/// This is needed for VCM MIS weight computation on bidirectional paths.
/// The function mirrors `brdf_pdf()` in pathtracer.wgsl but swaps the roles
/// of wi and wo when evaluating the GGX VNDF component.
fn brdf_pdf_reverse(wo: vec3<f32>, wi: vec3<f32>, ray_hit: ResolvedRayHitFull) -> f32 {
    if ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD {
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

    // Reverse: treat wi as the "viewing" direction, wo as the "sampled" direction
    // Diffuse PDF only depends on the sampled direction's cosine with N
    let diffuse_pdf = max(0.0, wo_tangent.z) / PI;
    // Specular VNDF PDF with wi as the incident (viewing) direction
    let specular_pdf = ggx_vndf_pdf(wi_tangent, wo_tangent, ray_hit.material.roughness);
    return diffuse_weight * diffuse_pdf + specular_weight * specular_pdf;
}

// ---------------------------------------------------------------------------
// Light emission PDF and sampling (Phase 6.4)
// ---------------------------------------------------------------------------

/// Result of sampling emission from a light source: a position on the light,
/// an outgoing emission direction, and the associated PDFs.
struct LightEmissionSample {
    /// World-space position on the light surface (area lights) or
    /// a sentinel value for directional lights.
    position: vec3<f32>,
    /// World-space normal at the emission point.
    normal: vec3<f32>,
    /// Outgoing emission direction (away from the light surface).
    direction: vec3<f32>,
    /// Emitted radiance along `direction`.
    radiance: vec3<f32>,
    /// PDF of choosing this position on this light source (w.r.t. area).
    /// For directional lights this is set to 1.0 (delta in direction).
    pdf_position: f32,
    /// PDF of the emission direction given the position (w.r.t. solid angle).
    /// For area lights this is cos(theta) / PI (cosine-weighted hemisphere).
    /// For directional lights this is 1.0 (delta distribution).
    pdf_direction: f32,
    /// Probability of picking this light among all light sources.
    pdf_light_pick: f32,
    /// True when the light source is a directional (infinite) light.
    is_directional: bool,
}

/// Computes the directional PDF of emission from an area light surface.
/// Given a surface normal and an outgoing direction, returns the
/// cosine-weighted hemisphere PDF: cos(theta) / PI. Returns 0 if the
/// direction points below the surface.
fn light_emission_direction_pdf(normal: vec3<f32>, direction: vec3<f32>) -> f32 {
    return max(0.0, dot(normal, direction)) / PI;
}

/// Computes the positional PDF of a point on an area light.
/// For uniform sampling over a triangle mesh with `triangle_count`
/// triangles each of area `triangle_area`, the PDF w.r.t. area is:
///   1.0 / (triangle_count * triangle_area)
fn light_emission_position_pdf(triangle_count: u32, triangle_area: f32) -> f32 {
    return 1.0 / (f32(triangle_count) * triangle_area);
}

/// Samples a light source and an emission point + direction from it.
/// This is used to start light subpaths in bidirectional methods.
///
/// For area lights (emissive meshes):
///   - Position: uniform random point on a random triangle
///   - Direction: cosine-weighted hemisphere above the surface normal
///
/// For directional lights:
///   - Position: not physically meaningful (set to vec3(0))
///   - Direction: the light direction (possibly jittered for soft shadows)
fn sample_light_emission(rng: ptr<function, u32>) -> LightEmissionSample {
    var result: LightEmissionSample;

    let light_count = arrayLength(&light_sources);
    let light_id = rand_range_u(light_count, rng);
    let light_source = light_sources[light_id];

    result.pdf_light_pick = 1.0 / f32(light_count);

    if light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
        let dir_light = directional_lights[light_source.id];

        result.position = vec3(0.0);
        result.normal = -dir_light.direction_to_light;
        result.direction = -dir_light.direction_to_light;
        result.radiance = dir_light.luminance;
        result.pdf_position = 1.0;
        result.pdf_direction = 1.0;
        result.is_directional = true;
    } else {
        // Emissive mesh: pick a random triangle, then a random point on it
        let triangle_count = light_source.kind >> 1u;
        let triangle_id = rand_range_u(triangle_count, rng);

        let seed = rand_u(rng);
        let barycentrics = triangle_barycentrics(seed);
        let triangle_data = resolve_triangle_data_full(light_source.id, triangle_id, barycentrics);

        result.position = triangle_data.world_position;
        result.normal = triangle_data.world_normal;
        result.radiance = triangle_data.material.emissive;

        // Positional PDF: 1 / (total surface area visible to this sampling strategy)
        result.pdf_position = 1.0 / (f32(triangle_count) * triangle_data.triangle_area);

        // Sample cosine-weighted hemisphere direction for emission
        result.direction = sample_cosine_hemisphere(triangle_data.world_normal, rng);

        // Directional PDF: cos(theta) / PI for cosine-weighted hemisphere
        let cos_theta = max(0.0, dot(result.direction, triangle_data.world_normal));
        result.pdf_direction = cos_theta / PI;

        result.is_directional = false;
    }

    return result;
}
