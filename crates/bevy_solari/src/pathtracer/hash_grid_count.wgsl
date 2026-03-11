#import bevy_solari::hash_grid::{hash_grid_index}
#import bevy_solari::light_vertex::{LightPathVertex, VcmParams}

/// Pass 1 of hash grid construction: Count.
///
/// Each light vertex computes its cell hash and atomically increments
/// the corresponding cell's count. This produces per-cell counts that
/// will be prefix-summed in the next pass.

@group(0) @binding(0) var<storage, read> light_vertices: array<LightPathVertex>;
@group(0) @binding(1) var<storage, read> light_vertex_count_ro: array<u32>;
@group(0) @binding(2) var<uniform> vcm_params: VcmParams;
@group(0) @binding(3) var<storage, read_write> cell_counts: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1)
fn hash_grid_count(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let vertex_index = global_id.x + global_id.y * num_workgroups.x * 64u;

    // Read the actual number of stored light vertices (written by light trace pass)
    let actual_count = light_vertex_count_ro[0];
    if vertex_index >= actual_count {
        return;
    }

    let cell_size_inv = 1.0 / (2.0 * vcm_params.merge_radius);
    let hash = hash_grid_index(light_vertices[vertex_index].position_and_perceptual_roughness.xyz, cell_size_inv);

    atomicAdd(&cell_counts[hash], 1u);
}
