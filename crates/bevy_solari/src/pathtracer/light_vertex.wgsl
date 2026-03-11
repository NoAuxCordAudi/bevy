#define_import_path bevy_solari::light_vertex

/// A vertex stored during light subpath tracing for later use in
/// vertex connections (BPT) and vertex merging (photon mapping).
///
/// Each non-specular hit along a light subpath produces one of these.
/// The dVCM/dVC/dVM partial weights are used by the VCM MIS framework
/// to correctly weight contributions from connections and merging.
///
/// Layout is packed to minimize wasted padding. Each vec3 is followed by
/// a useful scalar to fill the 16-byte vec4 slot. Fields unused by the
/// camera trace shader (emissive, path_length) have been removed.
///
/// Total size: 6 * vec4(16) + 2 * f32(4) = 104 bytes, padded to 112 for vec4 alignment.
struct LightPathVertex {
    /// World-space position of this vertex.
    /// .w = perceptual_roughness (packed to avoid padding waste).
    position_and_perceptual_roughness: vec4<f32>,
    /// Shading normal at this vertex.
    /// .w = metallic (packed).
    normal_and_metallic: vec4<f32>,
    /// Accumulated path throughput (radiance weight) arriving at this vertex.
    /// .w = roughness (linear, derived from perceptual_roughness).
    throughput_and_roughness: vec4<f32>,
    /// Base color of the material at this vertex.
    /// .w = reflectance (average of vec3 reflectance, sufficient for MIS).
    base_color_and_reflectance: vec4<f32>,
    /// Direction arriving at this vertex (pointing toward the previous vertex
    /// along the light subpath, i.e., the direction light travels *from*).
    /// .w = dVCM (MIS partial weight for vertex connection and merging).
    incoming_direction_and_dVCM: vec4<f32>,
    /// Tangent vector for TBN reconstruction (xyz = tangent, w = sign).
    world_tangent: vec4<f32>,
    /// MIS partial weight for vertex connections.
    dVC: f32,
    /// MIS partial weight for vertex merging.
    dVM: f32,
}

/// Uniform buffer containing VCM algorithm parameters shared between
/// the light trace and camera trace shaders.
struct VcmParams {
    /// Total number of light subpaths to trace (typically = num_pixels).
    num_light_paths: u32,
    /// Maximum number of light vertices that can be stored in the buffer.
    max_light_vertices: u32,
    /// Current merge radius for vertex merging.
    merge_radius: f32,
    /// PI * merge_radius^2 * num_light_paths — used in MIS weight computation.
    vm_weight: f32,
    /// 1.0 / vm_weight — used in MIS weight computation.
    vc_weight: f32,
    /// Current iteration number (for progressive radius reduction and RNG seeding).
    iteration: u32,
    /// Viewport width in pixels.
    screen_width: u32,
    /// Viewport height in pixels.
    screen_height: u32,
    /// Whether vertex connections (BPT) are enabled (1 = yes, 0 = no).
    enable_vc: u32,
    /// Whether vertex merging (photon mapping) is enabled (1 = yes, 0 = no).
    enable_vm: u32,
    /// Debug visualization mode:
    ///   0 = normal rendering (default)
    ///   1 = show dominant strategy per pixel (NEE=green, VC=blue, VM=red, BRDF=white, splat=yellow)
    ///   2 = show light vertex density heatmap at first camera hit
    debug_mode: u32,
    /// Padding to maintain 16-byte alignment.
    _pad_debug: u32,
}
