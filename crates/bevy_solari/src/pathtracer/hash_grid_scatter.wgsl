#import bevy_solari::hash_grid::{hash_grid_index}
#import bevy_solari::light_vertex::{LightPathVertex, VcmParams}

/// Pass 3 of hash grid construction: Scatter.
///
/// Each light vertex recomputes its cell hash and uses an atomic increment
/// on the (now prefix-summed) cell offsets to determine its position in the
/// sorted vertex indices array. The vertex's global index is written to that
/// position.
///
/// After this pass, the sorted_indices array contains light vertex indices
/// grouped by their hash cell, and the cell_offsets array has been consumed
/// (each entry now points past the last written index for that cell, which
/// equals cell_offsets_original[hash + 1]).

@group(0) @binding(0) var<storage, read> light_vertices: array<LightPathVertex>;
@group(0) @binding(1) var<storage, read> light_vertex_count_ro: array<u32>;
@group(0) @binding(2) var<uniform> vcm_params: VcmParams;
@group(0) @binding(3) var<storage, read_write> cell_offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sorted_indices: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn hash_grid_scatter(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let vertex_index = global_id.x + global_id.y * num_workgroups.x * 64u;

    // Read the actual number of stored light vertices
    let actual_count = light_vertex_count_ro[0];
    if vertex_index >= actual_count {
        return;
    }

    let cell_size_inv = 1.0 / (2.0 * vcm_params.merge_radius);
    let hash = hash_grid_index(light_vertices[vertex_index].position_and_perceptual_roughness.xyz, cell_size_inv);

    // Atomically claim a slot in the sorted array for this cell
    let slot = atomicAdd(&cell_offsets[hash], 1u);
    sorted_indices[slot] = vertex_index;
}
