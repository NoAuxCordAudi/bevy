#define_import_path bevy_solari::sky

#import bevy_solari::scene_bindings::directional_lights

// ── Time-of-day color presets ────────────────────────────────────────
// Each regime is defined by a sun_dir.y threshold.
// Night: y < -0.1 | Twilight: -0.1..0.0 | Sunset: 0.0..0.15 | Golden: 0.15..0.3 | Day: y > 0.3

// Night
const NIGHT_ZENITH:   vec3<f32> = vec3(0.005, 0.007, 0.02);
const NIGHT_HORIZON:  vec3<f32> = vec3(0.01, 0.01, 0.015);
const NIGHT_HAZE:     vec3<f32> = vec3(0.0, 0.0, 0.0);
const NIGHT_GLOW:     vec3<f32> = vec3(0.0, 0.0, 0.0);
const NIGHT_HAZE_POW: f32 = 8.0;
const NIGHT_GLOW_POW: f32 = 64.0;

// Twilight
const TWILIGHT_ZENITH:   vec3<f32> = vec3(0.03, 0.04, 0.1);
const TWILIGHT_HORIZON:  vec3<f32> = vec3(0.1, 0.05, 0.08);
const TWILIGHT_HAZE:     vec3<f32> = vec3(0.05, 0.02, 0.02);
const TWILIGHT_GLOW:     vec3<f32> = vec3(0.05, 0.02, 0.01);
const TWILIGHT_HAZE_POW: f32 = 8.0;
const TWILIGHT_GLOW_POW: f32 = 32.0;

// Sunset / Sunrise
const SUNSET_ZENITH:   vec3<f32> = vec3(0.15, 0.1, 0.2);
const SUNSET_HORIZON:  vec3<f32> = vec3(0.5, 0.25, 0.15);
const SUNSET_HAZE:     vec3<f32> = vec3(0.4, 0.15, 0.05);
const SUNSET_GLOW:     vec3<f32> = vec3(0.5, 0.2, 0.05);
const SUNSET_HAZE_POW: f32 = 4.0;
const SUNSET_GLOW_POW: f32 = 16.0;

// Golden Hour
const GOLDEN_ZENITH:   vec3<f32> = vec3(0.1, 0.15, 0.4);
const GOLDEN_HORIZON:  vec3<f32> = vec3(0.45, 0.4, 0.5);
const GOLDEN_HAZE:     vec3<f32> = vec3(0.25, 0.18, 0.1);
const GOLDEN_GLOW:     vec3<f32> = vec3(0.4, 0.25, 0.1);
const GOLDEN_HAZE_POW: f32 = 6.0;
const GOLDEN_GLOW_POW: f32 = 32.0;

// Day
const DAY_ZENITH:   vec3<f32> = vec3(0.1, 0.2, 0.5);
const DAY_HORIZON:  vec3<f32> = vec3(0.35, 0.45, 0.6);
const DAY_HAZE:     vec3<f32> = vec3(0.15, 0.12, 0.08);
const DAY_GLOW:     vec3<f32> = vec3(0.3, 0.2, 0.1);
const DAY_HAZE_POW: f32 = 8.0;
const DAY_GLOW_POW: f32 = 64.0;

// Minimum ambient brightness so night sky isn't pitch black (starlight/moonlight)
const NIGHT_AMBIENT_FLOOR: f32 = 0.006;

// ── Sky parameter interpolation ──────────────────────────────────────

struct SkyParams {
    zenith:    vec3<f32>,
    horizon:   vec3<f32>,
    haze:      vec3<f32>,
    glow:      vec3<f32>,
    haze_pow:  f32,
    glow_pow:  f32,
}

fn sky_params_from_elevation(sun_y: f32) -> SkyParams {
    // Blend factors between adjacent regimes using smoothstep
    let t_twilight = smoothstep(-0.1, 0.0, sun_y);   // night → twilight
    let t_sunset   = smoothstep(0.0, 0.15, sun_y);    // twilight → sunset
    let t_golden   = smoothstep(0.15, 0.3, sun_y);    // sunset → golden
    let t_day      = smoothstep(0.3, 0.5, sun_y);     // golden → day

    // Chain the interpolations: start at night, blend through each regime
    var p: SkyParams;

    // Night → Twilight
    p.zenith   = mix(NIGHT_ZENITH,   TWILIGHT_ZENITH,   t_twilight);
    p.horizon  = mix(NIGHT_HORIZON,  TWILIGHT_HORIZON,  t_twilight);
    p.haze     = mix(NIGHT_HAZE,     TWILIGHT_HAZE,     t_twilight);
    p.glow     = mix(NIGHT_GLOW,     TWILIGHT_GLOW,     t_twilight);
    p.haze_pow = mix(NIGHT_HAZE_POW, TWILIGHT_HAZE_POW, t_twilight);
    p.glow_pow = mix(NIGHT_GLOW_POW, TWILIGHT_GLOW_POW, t_twilight);

    // → Sunset
    p.zenith   = mix(p.zenith,   SUNSET_ZENITH,   t_sunset);
    p.horizon  = mix(p.horizon,  SUNSET_HORIZON,  t_sunset);
    p.haze     = mix(p.haze,     SUNSET_HAZE,     t_sunset);
    p.glow     = mix(p.glow,     SUNSET_GLOW,     t_sunset);
    p.haze_pow = mix(p.haze_pow, SUNSET_HAZE_POW, t_sunset);
    p.glow_pow = mix(p.glow_pow, SUNSET_GLOW_POW, t_sunset);

    // → Golden Hour
    p.zenith   = mix(p.zenith,   GOLDEN_ZENITH,   t_golden);
    p.horizon  = mix(p.horizon,  GOLDEN_HORIZON,  t_golden);
    p.haze     = mix(p.haze,     GOLDEN_HAZE,     t_golden);
    p.glow     = mix(p.glow,     GOLDEN_GLOW,     t_golden);
    p.haze_pow = mix(p.haze_pow, GOLDEN_HAZE_POW, t_golden);
    p.glow_pow = mix(p.glow_pow, GOLDEN_GLOW_POW, t_golden);

    // → Day
    p.zenith   = mix(p.zenith,   DAY_ZENITH,   t_day);
    p.horizon  = mix(p.horizon,  DAY_HORIZON,  t_day);
    p.haze     = mix(p.haze,     DAY_HAZE,     t_day);
    p.glow     = mix(p.glow,     DAY_GLOW,     t_day);
    p.haze_pow = mix(p.haze_pow, DAY_HAZE_POW, t_day);
    p.glow_pow = mix(p.glow_pow, DAY_GLOW_POW, t_day);

    return p;
}

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

    // Effective brightness: never fully zero so night sky has some ambient
    let effective_brightness = max(sun_brightness, NIGHT_AMBIENT_FLOOR);

    let p = sky_params_from_elevation(sun_dir.y);

    let y = ray_direction.y;

    // Below horizon: exponential falloff to ground color tinted by time of day
    if y < 0.0 {
        let ground_tint = mix(p.horizon, p.zenith, 0.5);
        return ground_tint * effective_brightness * 0.04 * exp(y * 10.0);
    }

    // Zenith-to-horizon gradient
    let horizon_factor = 1.0 - y;
    var sky = mix(p.zenith, p.horizon, horizon_factor * horizon_factor);

    // Atmospheric haze brightening near horizon
    sky += p.haze * pow(horizon_factor, p.haze_pow);

    // Forward-scattering glow around the sun (Mie-like)
    let sun_cos = dot(ray_direction, sun_dir);
    if sun_cos > 0.0 {
        sky += p.glow * pow(sun_cos, p.glow_pow);
        // Broader secondary glow
        sky += p.glow * 0.33 * pow(sun_cos, max(p.glow_pow * 0.125, 2.0));
    }

    return sky * effective_brightness;
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
