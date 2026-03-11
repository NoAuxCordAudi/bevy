#define_import_path bevy_solari::sky

#import bevy_solari::scene_bindings::directional_lights

/// Evaluate ambient sky radiance (gradient, haze, forward-scattering glow)
/// WITHOUT the sun disk. Used by the pathtracer so the sun disk can be
/// MIS-weighted separately.
fn evaluate_sky_ambient(ray_direction: vec3<f32>) -> vec3<f32> {
    let light_count = arrayLength(&directional_lights);
    if light_count == 0u {
        return vec3(0.0);
    }

    let sun = directional_lights[0];
    let sun_dir = sun.direction_to_light;
    let sun_brightness = dot(sun.luminance * sun.inverse_pdf, vec3(1.0 / 3.0));

    let y = ray_direction.y;

    // Below horizon: quick exponential falloff to dark ground
    if y < 0.0 {
        return vec3(0.04, 0.04, 0.035) * sun_brightness * 0.02 * exp(y * 10.0);
    }

    // Zenith-to-horizon gradient
    let horizon_factor = 1.0 - y;
    let zenith_color = vec3(0.1, 0.2, 0.5);
    let horizon_color = vec3(0.35, 0.45, 0.6);
    var sky = mix(zenith_color, horizon_color, horizon_factor * horizon_factor);

    // Atmospheric haze brightening near horizon
    sky += vec3(0.15, 0.12, 0.08) * pow(horizon_factor, 8.0);

    // Forward-scattering glow around the sun (Mie-like)
    let sun_cos = dot(ray_direction, sun_dir);
    if sun_cos > 0.0 {
        sky += vec3(0.3, 0.2, 0.1) * pow(sun_cos, 64.0);
        sky += vec3(0.1, 0.08, 0.04) * pow(sun_cos, 8.0);
    }

    return sky * sun_brightness;
}

/// Evaluate sun disk radiance for a given ray direction.
/// Returns the sun's luminance if the ray hits the sun disk, zero otherwise.
fn evaluate_sun_disk(ray_direction: vec3<f32>) -> vec3<f32> {
    let light_count = arrayLength(&directional_lights);
    if light_count == 0u {
        return vec3(0.0);
    }

    let sun = directional_lights[0];
    let sun_cos = dot(ray_direction, sun.direction_to_light);
    if sun_cos > sun.cos_theta_max {
        return sun.luminance;
    }
    return vec3(0.0);
}

/// Combined sky radiance (ambient + sun disk). Used by specular_gi and other
/// paths that don't need separate sun MIS weighting.
fn evaluate_sky(ray_direction: vec3<f32>) -> vec3<f32> {
    return evaluate_sky_ambient(ray_direction) + evaluate_sun_disk(ray_direction);
}
