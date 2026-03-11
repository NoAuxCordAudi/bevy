#define_import_path bevy_solari::hash_grid

/// Spatial hash grid for efficient radius queries over light path vertices.
///
/// The grid uses a hash table of fixed size (power of 2) where each cell
/// covers a cube of side length `cell_size = 2 * merge_radius`. This ensures
/// that any two points within `merge_radius` of each other are either in the
/// same cell or in adjacent cells, so querying the 3x3x3 neighborhood of
/// cells around a point is sufficient to find all vertices within range.
///
/// The grid is built in three passes:
///   1. Count: each light vertex atomically increments its cell's count
///   2. Prefix sum: exclusive scan over cell counts to produce offsets
///   3. Scatter: each light vertex writes its index into the sorted array
///
/// After construction, querying iterates over the 27 neighboring cells and
/// checks distance for each candidate vertex.
///
/// ## Post-Scatter Buffer Layout
///
/// After the scatter pass, the cell_offsets buffer has been consumed:
/// `cell_offsets[i]` now equals the *end* offset of cell i (which is the
/// original start offset + count for that cell). This means:
///   - Start of cell `hash`: `if hash == 0 { 0 } else { cell_offsets[hash - 1] }`
///   - End of cell `hash`: `cell_offsets[hash]`
///
/// The query pattern in the camera shader should iterate over 27 neighboring
/// cells (3x3x3 neighborhood), and for each cell look up its range using
/// the above convention, then iterate over candidate vertices and distance-check.

/// Number of cells in the hash table. Must be a power of 2.
/// 131072 = 2^17. Using a mask for modular arithmetic.
const HASH_GRID_TABLE_SIZE: u32 = 131072u;
const HASH_GRID_TABLE_MASK: u32 = 131071u; // TABLE_SIZE - 1

/// Computes the integer cell coordinates for a world-space position.
fn hash_grid_cell(position: vec3<f32>, cell_size_inv: f32) -> vec3<i32> {
    return vec3<i32>(floor(position * cell_size_inv));
}

/// Hashes 3D integer cell coordinates to a table index.
/// Uses three large primes and bitwise XOR for spatial hashing.
fn hash_grid_hash(cell: vec3<i32>) -> u32 {
    let h = u32(cell.x) * 73856093u
          ^ u32(cell.y) * 19349663u
          ^ u32(cell.z) * 83492791u;
    return h & HASH_GRID_TABLE_MASK;
}

/// Computes the hash table index for a world-space position.
fn hash_grid_index(position: vec3<f32>, cell_size_inv: f32) -> u32 {
    return hash_grid_hash(hash_grid_cell(position, cell_size_inv));
}

/// Looks up the start and end indices in the sorted indices array for a
/// given cell hash. Uses the post-scatter convention where cell_offsets[hash]
/// is the *end* of the range.
///
/// Returns: vec2<u32>(start, end) — the range [start, end) in sorted_indices.
///
/// Note: This function cannot access the buffer directly since WGSL does not
/// support passing storage buffer references. Callers should inline this
/// pattern:
///   let end_ = cell_offsets[hash];
///   let start = select(cell_offsets[hash - 1u], 0u, hash == 0u);
///   // iterate sorted_indices[start..end_]
