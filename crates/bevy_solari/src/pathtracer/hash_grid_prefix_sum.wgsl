/// Pass 2 of hash grid construction: Prefix Sum (exclusive scan).
///
/// Performs an exclusive prefix sum over the cell counts array, converting
/// per-cell counts into per-cell offsets. After this pass, cell_counts[i]
/// contains the sum of all counts for cells 0..i-1 (i.e., the starting
/// offset into the sorted vertex indices array for cell i).
///
/// This uses a simple sequential scan in a single thread. For our table size
/// of 131072 entries, this completes in microseconds on a GPU and avoids
/// the complexity of a parallel prefix sum. Can be optimized later if needed.
///
/// The total count (sum of all cells) is written to cell_counts[TABLE_SIZE],
/// which serves as the sentinel for the last cell's count computation.
/// Therefore, the cell_counts buffer must be allocated with TABLE_SIZE + 1 entries.

/// Must match HASH_GRID_TABLE_SIZE in hash_grid.wgsl.
const TABLE_SIZE: u32 = 131072u;

@group(0) @binding(0) var<storage, read_write> cell_counts: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn hash_grid_prefix_sum() {
    var running_sum = 0u;
    for (var i = 0u; i < TABLE_SIZE; i += 1u) {
        let count = cell_counts[i];
        cell_counts[i] = running_sum;
        running_sum += count;
    }
    // Write the total as the sentinel after the last cell.
    // This allows cell count lookup as: count[i] = offsets[i+1] - offsets[i]
    cell_counts[TABLE_SIZE] = running_sum;
}
