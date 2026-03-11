# Migration Plan: Unidirectional Path Tracing → Vertex Connection and Merging (VCM)

## References

- [Light Transport Simulation with Vertex Connection and Merging](https://cgg.mff.cuni.cz/~jaroslav/papers/2012-vcm/2012-vcm-paper.pdf) — Georgiev, Křivánek, Davidovič, Slusallek (SIGGRAPH Asia 2012)
- [SmallVCM Reference Implementation](http://www.smallvcm.com/) — Educational C++ implementation
- [SmallVCM Source (GitHub)](https://github.com/SmallVCM/SmallVCM)
- [VCM Walkthrough — Joe Schutte](https://schuttejoe.github.io/post/vertexconnectionandmerging/)
- [Lumen — Vulkan RT Framework with VCM](https://github.com/yuphin/Lumen)
- [GPU-VCM/BDPM — GPU Path Tracer with VCM](https://github.com/GPU-VCM/BDPM)

---

## Current Architecture Summary

The existing pathtracer in `bevy_solari` is a **unidirectional forward path tracer** with:

- Camera-to-scene ray tracing with up to 128 bounces
- Next-Event Estimation (NEE) for direct lighting via shadow rays
- Multiple Importance Sampling (MIS) between BRDF sampling and light sampling (power heuristic)
- Importance-sampled BRDF: cosine-weighted diffuse + GGX VNDF specular
- Hardware ray queries against TLAS/BLAS acceleration structures
- Progressive accumulation with Welford's online variance for convergence testing
- Halton sequence sub-pixel jittering

### Key Files

| File | Role |
|------|------|
| `pathtracer.wgsl` | Main GPU compute shader (ray generation, path tracing loop, accumulation) |
| `node.rs` | Pipeline creation and compute dispatch |
| `mod.rs` | Plugin definition, `Pathtracer` component |
| `prepare.rs` | Accumulation/variance texture allocation |
| `extract.rs` | Camera data extraction |
| `scene/brdf.wgsl` | Lambertian diffuse + GGX microfacet specular BRDF |
| `scene/sampling.wgsl` | Light sampling, BRDF importance sampling, MIS |
| `scene/raytracing_scene_bindings.wgsl` | Ray tracing infrastructure, hit resolution |
| `scene/binder.rs` | TLAS construction, material/light binding |
| `scene/blas.rs` | BLAS management and compaction |

---

## VCM Algorithm Overview

VCM unifies **Bidirectional Path Tracing (BPT)** and **Progressive Photon Mapping (PPM)** under a single MIS framework. It traces paths from both the camera and light sources, then combines contributions via two mechanisms:

1. **Vertex Connection (VC)** — deterministic connections between camera and light subpath vertices (from BPT)
2. **Vertex Merging (VM)** — density estimation by merging nearby light vertices within a search radius (from photon mapping)

### Three-Phase Pipeline

1. **Light subpath tracing** — trace paths from light sources, store non-specular vertices
2. **Hash grid construction** — spatial index over stored light vertices for radius queries
3. **Camera subpath tracing** — trace camera paths, perform connections + merging at each vertex

---

## Migration Task List

### Phase 1: Foundation — Light Subpath Infrastructure

#### 1.1 Light Subpath Storage Buffer
- [x] Define a `LightPathVertex` GPU struct in `light_vertex.wgsl` containing:
  - `world_position: vec3<f32>`, `world_normal: vec3<f32>`, `throughput: vec3<f32>`
  - Material subset: `base_color`, `emissive`, `perceptual_roughness`, `metallic`, `reflectance`, `roughness`
  - `incoming_direction: vec3<f32>`, `world_tangent: vec4<f32>` (for TBN reconstruction)
  - `path_length: u32`, `dVCM: f32`, `dVC: f32`, `dVM: f32`
- [x] Define `VcmParams` uniform struct in `light_vertex.wgsl`
- [x] Allocate GPU storage buffer sized for `num_pixels * max_light_path_length` vertices — in `prepare.rs` (`LightVertexBuffer`)
- [x] Add a light vertex count buffer (atomic counter) — in `prepare.rs` (`LightVertexCountBuffer`)
- [x] Add VCM params uniform buffer — in `prepare.rs` (`VcmParamsBuffer`)
- [x] Register these buffers in the bind group layout — in `node.rs` (`light_trace_bind_group_layout`)

#### 1.2 Light Subpath Tracing Compute Shader
- [x] Create `light_trace.wgsl` — new compute shader for light subpath generation
- [x] Implement light emission sampling via `sample_light_emission()` from `sampling.wgsl`
- [x] Initialize `dVCM`, `dVC`, `dVM` from emission PDFs
- [x] Path tracing loop with:
  - Vertex storage at non-specular hits (atomic increment counter)
  - BRDF importance sampling for next direction (duplicated from pathtracer.wgsl as `importance_sample_bounce`)
  - VCM MIS weight updates using forward/reverse PDFs via `evaluate_brdf_bidirectional()`
  - Specular surface handling (zero dVC/dVM on delta BSDF)
  - Russian roulette for termination (after 3 bounces)
  - Geometry term conversions for dVCM/dVC/dVM at each vertex arrival
- [x] Camera connection / light tracing contribution (implemented in Phase 7)
- [x] Add render graph node for light tracing dispatch — in `node.rs`

#### 1.3 Render Graph Node for Light Tracing
- [x] Created light trace compute pipeline in `init_pathtracer_pipelines`
- [x] Light trace dispatched before camera trace in `pathtracer` system
- [x] Light vertex counter cleared to zero each frame via `clear_buffer`
- [x] Added `vcm_iteration` counter to `Pathtracer` component, incremented in `extract.rs`
- [x] Progressive merge radius: `initial_merge_radius / sqrt(iteration + 1)`

#### 1.3 Reverse PDF Infrastructure
- [x] Extend `brdf.wgsl` with `evaluate_brdf_bidirectional()` returning forward+reverse PDFs — completed in Phase 6
- [x] For Lambertian: reverse PDF = `cos(theta_out) / PI` (cosine of the "sampled" direction)
- [x] For GGX specular: reverse PDF via VNDF with swapped in/out directions — `ggx_vndf_pdf_internal(wi, wo)` and `ggx_vndf_pdf(wi, wo)` in sampling.wgsl
- [x] `brdf_pdf_reverse()` standalone function in `sampling.wgsl` for cases where only the reverse PDF is needed
- [x] `evaluate_brdf_bidirectional()` in `brdf.wgsl` computes both forward and reverse PDFs in a single call

---

### Phase 2: Spatial Hash Grid for Vertex Merging

#### 2.1 Hash Grid Data Structure
- [x] Create `hash_grid.wgsl` — GPU hash grid library with spatial hashing utilities
  - `hash_grid_cell()`: world position to integer cell coordinates
  - `hash_grid_hash()`: integer cell coordinates to table index via 3 large primes + XOR
  - `hash_grid_index()`: combined position-to-hash convenience function
  - Constants: `HASH_GRID_TABLE_SIZE = 131072` (2^17), `HASH_GRID_TABLE_MASK = 131071`
  - Cell size = `2 * merge_radius`, cell_size_inv computed in each shader
- [x] Allocate two GPU buffers in `prepare.rs`:
  - **Cell counts/offsets** buffer (`HashGridCellBuffer`) — `u32[TABLE_SIZE + 1]`, +1 for sentinel
  - **Sorted vertex indices** buffer (`HashGridSortedIndicesBuffer`) — `u32[max_light_vertices]`
  - Both have `STORAGE | COPY_DST` usage (cell buffer needs clearing each frame)
- [x] Implement three-pass construction with separate compute shaders:
  1. **Count pass** (`hash_grid_count.wgsl`): each light vertex atomically increments its cell's count
  2. **Prefix sum** (`hash_grid_prefix_sum.wgsl`): sequential exclusive scan in a single thread (TABLE_SIZE=131072 entries, fast enough)
  3. **Scatter pass** (`hash_grid_scatter.wgsl`): each light vertex atomically claims a slot and writes its index
- [x] Registered all shaders in `mod.rs` (`embedded_asset!` for compute shaders, `load_shader_library!` for hash_grid.wgsl)
- [x] Created compute pipelines and bind group layouts in `node.rs` (`init_pathtracer_pipelines`)
- [x] Added dispatch sequence in `pathtracer` system: light_trace -> clear_hash_grid -> hash_count -> hash_prefix_sum -> hash_scatter -> camera_trace
- [x] Each pass runs in its own compute pass for automatic storage buffer barriers

#### 2.2 Hash Grid Query
- [x] Documented query pattern in `hash_grid.wgsl` for use by camera trace shader (Phase 3)
- [x] Post-scatter buffer convention: `cell_offsets[hash]` = end of range, start = `cell_offsets[hash-1]` or 0
- [ ] Actual query loop implementation deferred to Phase 3 (camera subpath modifications)
  - Will query 27 neighboring cells (3x3x3 around query point)
  - Distance check: `distance^2 < merge_radius^2` for each candidate

#### 2.3 Merge Radius Management
- [x] Already implemented in Phase 1: `merge_radius = initial_merge_radius / sqrt(iteration + 1)` in `prepare.rs`
- [x] `initial_merge_radius` field on `Pathtracer` component (default: 0.1 world units)
- [x] `merge_radius` stored in `VcmParams` uniform, accessible to all shaders
- [x] Hash grid cell_size derives from merge_radius: `cell_size = 2.0 * merge_radius`
- [ ] Scene-relative radius (fraction of scene diameter) — deferred to tuning phase

---

### Phase 3: Camera Subpath Modifications

#### 3.1 Bidirectional MIS Tracking on Camera Paths
- [x] Add `dVCM`, `dVC`, `dVM` tracking to the camera path loop in `pathtracer.wgsl`
- [x] Initialize camera path MIS terms:
  ```
  dVCM = num_light_paths (pinhole camera: camera_pdf_area = 1.0)
  dVC  = 0
  dVM  = 0
  ```
- [x] Update terms at each bounce (same recursive formulas as light paths, with camera-side PDFs)
- [x] Geometry term conversion at vertex arrival: `dVCM *= dist^2/cos_theta`, `dVC /= cos_theta`, `dVM /= cos_theta`
- [x] Specular handling: zero `dVC`/`dVM` on specular bounces

#### 3.2 Vertex Connection (VC) — Deterministic Connections
- [x] At each non-specular camera vertex, connect to light subpath vertices via `vertex_connection()`:
  - Select `MAX_LIGHT_PATH_LENGTH` (10) light vertices using deterministic hash of pixel_index
  - Evaluate BRDF at both endpoints using `evaluate_brdf()` and `evaluate_brdf_bidirectional()`
  - Compute geometric coupling (cos_camera * cos_light factored into BRDF, 1/dist^2 for area conversion)
  - Test visibility with `trace_point_visibility()` shadow ray
  - Compute VCM MIS weight:
    ```
    w_light = pdf_rev_camera * (cam_dVCM + cam_dVC * pdf_fwd_camera)
    w_camera = pdf_rev_light * (lv.dVCM + lv.dVC * pdf_fwd_light)
    mis_weight = 1.0 / (w_light + 1.0 + w_camera)
    ```
  - Scale contribution by `total_count / num_connections` to compensate for subsampling
- [x] Guard: skip connections when either vertex is specular (`is_specular_surface()`)

#### 3.3 Vertex Merging (VM) — Photon Density Estimation
- [x] At each non-specular camera vertex, query hash grid for nearby light vertices via `vertex_merging()`
- [x] 27-cell (3x3x3) neighborhood query using `hash_grid_cell()` and `hash_grid_hash()`
- [x] Post-scatter buffer convention: `cell_offsets[hash]` = end, `cell_offsets[hash-1]` = start (or 0)
- [x] For each light vertex within `merge_radius`:
  - Distance check: `dist^2 < merge_radius^2`
  - Evaluate BRDF at camera vertex using light vertex's incoming direction
  - Compute VCM MIS weight for merging:
    ```
    w_light = pdf_rev_camera * (cam_dVCM + cam_dVM * pdf_fwd_camera)
    w_camera = lv.dVCM * vc_weight + lv.dVM * pdf_rev_light + 1.0
    mis_weight = 1.0 / (w_light + 1.0 + w_camera)
    ```
  - Kernel normalization: `1.0 / (PI * merge_radius^2 * num_light_paths)`
- [x] Skip specular light vertices

#### 3.4 Retain Existing NEE (Connection to Light Source Directly)
- [x] Keep the existing direct light sampling (NEE) as the `s=0` connection strategy
- [ ] Modify its MIS weight computation to account for the new VCM MIS framework (incorporate `dVCM` terms) — deferred to Phase 4
- [x] Keep the existing emissive-hit contribution as the `t=0` camera strategy with existing MIS weights

#### 3.5 Bind Group Changes
- [x] Added 5 new read-only bindings to camera trace bind group layout (indices 5-9):
  - @binding(5): light_vertices (read-only storage)
  - @binding(6): light_vertex_count (read-only storage)
  - @binding(7): hash_grid_cell_offsets (read-only storage)
  - @binding(8): hash_grid_sorted_indices (read-only storage)
  - @binding(9): vcm_params (uniform)
- [x] Updated `init_pathtracer_pipelines()` bind group layout in `node.rs`
- [x] Updated camera trace bind group creation in `pathtracer()` system to pass all 10 bindings
- [x] Helper functions: `material_from_light_vertex()`, `brdf_pdf_at_light_vertex()` for working with light vertex data

---

### Phase 4: MIS Weight Balancing Constants

#### 4.1 VC/VM Balancing
- [x] Compute per-frame balancing constants:
  ```
  vm_count = num_light_paths  (total light subpaths)
  vc_count = num_light_paths  (typically equal)
  vm_weight = PI * merge_radius² * vm_count
  vc_weight = 1.0 / vm_weight
  ```
- [x] Pass these as uniforms to both light and camera shaders — in `prepare.rs`, uploaded via `VcmParamsUniform`
- [x] These constants control the relative contribution of connections vs. merging — scene-dependent tuning may be needed
- [x] Verified formulas match SmallVCM: `vm_weight = PI * r^2 * N` and `vc_weight = 1/vm_weight` in `prepare.rs`

#### 4.2 NEE MIS Weight Update for VCM
- [ ] Modify NEE's MIS weight computation to use VCM framework (incorporate dVCM terms) — deferred, current power_heuristic approach works for direct lighting

#### 4.3 MIS Weight Validation
- [x] Implement debug visualization: color-code pixels by dominant strategy (NEE, VC, VM, BRDF sampling) — implemented as `debug_mode = 1` on `Pathtracer` component
- [ ] Verify weights sum to 1.0 across all strategies for a given path length — requires GPU runtime testing
- [ ] Test on known scenes: Cornell box with caustics (ring on floor), SDS paths (light→specular→diffuse→specular→camera) — requires GPU runtime testing

---

### Phase 5: Render Graph & Pipeline Integration

#### 5.1 Multi-Pass Render Graph
- [x] Restructure the render graph to execute in order:
  1. **Light trace pass** — dispatch `light_trace.wgsl` (one thread per light subpath)
  2. **Hash grid build pass** — count, prefix sum, scatter (3 dispatches)
  3. **Camera trace pass** — dispatch modified `pathtracer.wgsl` (one thread per pixel)
- [x] Add proper buffer barriers between passes:
  - Each sub-pass runs in its own compute pass for automatic storage buffer barriers
- [x] Reset light vertex counter and hash grid each frame via `clear_buffer`

#### 5.2 Resource Management
- [x] Update `prepare.rs` to allocate:
  - Light vertex buffer (sized per `max_light_paths * max_path_length`)
  - Hash grid cell count buffer
  - Hash grid vertex index buffer
  - Atomic counter buffer
- [x] Add buffer resizing logic when resolution or settings change
  - `VcmBufferViewportSize` component tracks last-allocated viewport dimensions
  - Size-dependent buffers (light vertices, hash grid sorted indices) only reallocated on viewport change
  - VCM params uniform updated every frame (iteration/merge_radius change per frame)
- [x] Update bind group creation in `node.rs` to include all new buffers

#### 5.3 Settings & Configuration
- [x] Added `enable_vc: bool` and `enable_vm: bool` to `Pathtracer` component (default: true)
- [x] Added `enable_vc: u32` and `enable_vm: u32` to `VcmParamsUniform` (Rust) and `VcmParams` (WGSL)
- [x] Runtime toggles in `pathtracer.wgsl`: VC and VM calls guarded by `vcm_params.enable_vc != 0u` / `vcm_params.enable_vm != 0u`
- [x] Settings uploaded via `VcmParamsUniform` uniform buffer each frame
- [ ] Additional settings (max_path_length, radius_alpha, light_paths_per_pixel) deferred to tuning phase

---

### Phase 6: BRDF & Material System Extensions

#### 6.1 Specular Surface Handling
- [x] Classify surfaces as specular (roughness ≤ 0.001) vs. non-specular — `is_specular_surface()` in `brdf.wgsl`
- [x] Skip vertex storage, connections, and merging on specular surfaces (only BRDF sampling applies) — `is_specular_surface()` returns true when roughness ≤ MIRROR_ROUGHNESS_THRESHOLD; `evaluate_brdf_bidirectional()` returns pdf_forward=pdf_reverse=0 for specular surfaces
- [x] Ensure MIS weight formulas handle specular-to-diffuse transitions correctly (delta BSDF PDFs → infinite, handled by zeroing `dVC`/`dVM` on specular bounces) — callers should check `is_specular_surface()` and zero dVC/dVM

#### 6.2 Combined Forward/Reverse BRDF Evaluation
- [x] Create `evaluate_brdf_bidirectional()` that returns:
  - BRDF value `f(wi, wo)`
  - Forward PDF `p(wi | wo)` — probability of sampling wi given viewing direction wo
  - Reverse PDF `p(wo | wi)` — probability of sampling wo given viewing direction wi
- [x] This avoids redundant half-vector and GGX computations — uses `ggx_vndf_pdf_internal()` in `brdf.wgsl`
- [x] Update `brdf.wgsl` with this combined function

#### 6.3 Light Source Emission Profiles
- [x] Implement `light_emission_direction_pdf(normal, direction)` and `light_emission_position_pdf(triangle_count, triangle_area)` for:
  - Area lights (emissive meshes): uniform over surface, cosine-weighted direction
  - Directional lights: delta in direction, handled with pdf=1.0
- [x] Implement `sample_light_emission()` returning position, direction, and PDFs — in `sampling.wgsl`

#### 6.4 Reverse PDF Infrastructure (moved from Phase 1.3)
- [x] `brdf_pdf_reverse()` in `sampling.wgsl` — swaps wi/wo roles for VNDF evaluation
- [x] `evaluate_brdf_bidirectional()` in `brdf.wgsl` computes both forward and reverse PDFs in a single call

---

### Phase 7: Camera Connection (Light Tracing Contribution)

#### 7.1 Light-to-Camera Direct Connection
- [x] During light subpath tracing, at each non-specular vertex:
  - Compute direction to camera
  - Test visibility with shadow ray to camera
  - Evaluate BRDF at light vertex toward camera
  - Project onto image plane to determine pixel coordinates (using `clip_from_world` matrix)
  - Compute MIS weight using light path's `dVCM`/`dVC` terms
- [x] Splat contribution to the correct pixel via atomic fixed-point splatting buffer
- [x] Handle off-screen projections (clip.w <= 0, NDC bounds check) and near-plane clipping
- [x] Camera forward direction extracted from `world_from_view[2]` column
- [x] Camera PDF computed as `f_x * f_y * W * H / (4 * cos^3(theta))` for pinhole model
- [x] MIS weight: `1 / (pdf_camera_A * (dVCM + dVC * num_light_paths) + 1)`

#### 7.2 Atomic Pixel Splatting
- [x] Uses `array<atomic<u32>>` buffer with 3 u32s per pixel (R, G, B channels)
- [x] Fixed-point encoding: float * 65536 (2^16), accumulated via `atomicAdd`
- [x] Camera trace reads back via `read_splat()`: u32 values * (1/65536) to recover float
- [x] Splat buffer allocated in `prepare.rs` (`SplatBuffer` component)
- [x] Splat buffer cleared to zero each frame via `clear_buffer` in `node.rs`

#### 7.3 Compositing
- [x] Camera trace shader reads splatted contributions before exposure and accumulation
- [x] Exposure is applied uniformly to all radiance (camera path + splatted light tracing)
- [x] No separate compositing pass needed — integrated into the camera trace shader

#### 7.4 Bind Group Changes
- [x] Light trace bind group extended: added view uniform (@binding(3)) and splat buffer (@binding(4))
- [x] Camera trace bind group extended: added splat buffer (@binding(10), read-only)
- [x] View uniform uses `.buffer()` instead of `.binding()` for reuse across both bind groups

---

### Phase 8: Accumulation & Convergence Updates

#### 8.1 Progressive Accumulation Modifications
- [x] VCM is inherently progressive — each iteration produces a full image estimate
- [x] Running average `mix(old_color.rgb, radiance, 1.0 / (old_color.a + 1.0))` already works correctly
- [x] The merge radius reduction makes this a *progressive* algorithm (biased per-iteration, consistent in the limit)
- [x] Verified all VCM contributions are included in `radiance` before accumulation:
  - Camera path (emissive hits + NEE): accumulated via throughput in main loop
  - Vertex connection (BPT): added to `radiance` at each non-specular bounce
  - Vertex merging (photon density estimation): added to `radiance` at each non-specular bounce
  - Splat buffer (light tracing camera connections): composited via `read_splat()` before accumulation
- [x] Exposure applied uniformly to all radiance sources before accumulation

#### 8.2 Variance Tracking Updates
- [x] Existing Welford's online variance on luminance works without modification
- [x] Variance tracks total radiance regardless of source (camera path, VC, VM, splat)
- [x] VM contributions may cause higher initial variance but converge faster for caustics — no special handling needed
- [ ] Per-strategy variance tracking for diagnostics — deferred to Phase 10 (testing)

#### 8.3 Convergence Criteria
- [x] Existing dual convergence test (relative + absolute floor) retained without modification
- [x] Default `min_samples = 64` is already reasonable for VCM (provides enough iterations for merge radius to shrink and noise characteristics to stabilize)
- [ ] Adaptive merge radius based on per-pixel variance — deferred to Phase 9 (optimization)

---

### Phase 9: Optimization

#### 9.1 Memory Optimization
- [x] Packed `LightPathVertex` struct: removed unused fields (`emissive`, `path_length`) and explicit padding (`_pad0/1/2`)
  - Each vec3 is packed with a useful scalar into a vec4 (e.g., `position_and_perceptual_roughness`, `normal_and_metallic`)
  - Reduced struct size from 176 bytes (over-allocated) to 112 bytes (6 vec4s + 2 f32s, padded to 16-byte alignment)
  - ~36% memory reduction for light vertex buffer
- [x] Updated all consumers: `light_trace.wgsl`, `pathtracer.wgsl` (vertex_connection, vertex_merging, material_from_light_vertex, brdf_pdf_at_light_vertex), `hash_grid_count.wgsl`, `hash_grid_scatter.wgsl`
- [x] Updated `LIGHT_VERTEX_SIZE_BYTES` in `prepare.rs` to match new struct size (112 bytes)
- [ ] Consider `f16` for throughput and normals — deferred (precision risk vs. moderate savings)
- [ ] Reservoir sampling for fixed vertex budget — deferred to tuning phase

#### 9.2 Hash Grid Optimization
- [x] Hash table size of 131072 (2^17) is reasonable for most scenes — no changes needed
- [x] 27-cell query is simple and correct — no changes needed
- [ ] Multi-level hash grids for varying density — deferred
- [ ] Profile cell query performance — deferred

#### 9.3 Coherence & Occupancy
- [x] Vertex connections limited to `MAX_LIGHT_PATH_LENGTH` (10) per camera vertex — already reasonable
- [ ] Light path sorting for TLAS coherence — deferred
- [ ] Wavefront architecture — deferred
- [ ] Importance-based connection selection — deferred

#### 9.4 NaN/Infinity Guards
- [x] Added `is_valid_radiance()` helper: checks for NaN, Inf, and negative values
- [x] Added `sanitize_mis()` helper: returns 0.0 for NaN, Inf, or negative scalar values
- [x] Both helpers added to `light_trace.wgsl` and `pathtracer.wgsl`
- [x] MIS weight sanitization at all critical points:
  - Geometry term conversion at vertex arrival (both camera and light paths): guard against grazing angles (cos_theta < 1e-6)
  - MIS weight recursive update at bounce (both camera and light paths): sanitize_mis on all three d-values
  - Vertex connection MIS weight: sanitize_mis on w_light/w_camera, clamp final weight to [0, 1]
  - Vertex merging MIS weight: sanitize_mis on w_light/w_camera, clamp final weight to [0, 1]
  - Camera connection (light tracing): sanitize_mis on w_light, clamp final weight to [0, 1]
  - Light vertex storage: sanitize_mis on dVCM/dVC/dVM before writing to buffer
- [x] All contribution accumulation checks use `is_valid_radiance()` (rejects NaN, Inf, negative)

#### 9.5 Max Light Path Length Constant
- [x] Already defined: `MAX_LIGHT_BOUNCES = 10u` in `light_trace.wgsl`
- [x] Matching constant: `MAX_LIGHT_PATH_LENGTH = 10u` in `pathtracer.wgsl`
- [x] Matching Rust constant: `MAX_LIGHT_PATH_LENGTH = 10` in `prepare.rs`
- [x] Removed unused `path_length` tracking variable from `light_trace.wgsl` (was incremented but never read after struct packing removed the field)

#### 9.6 Denoising Compatibility
- [ ] VCM produces different noise characteristics than unidirectional PT — verify denoiser compatibility
- [ ] Output auxiliary buffers (albedo, normals, depth) for denoiser input as currently done

---

### Phase 10: Testing & Validation

#### 10.0 Debug Visualization (Code-only tooling)
- [x] Added `debug_mode: u32` to `Pathtracer` component, `VcmParamsUniform`, and `VcmParams` WGSL struct
  - 0 = normal rendering (default, zero overhead)
  - 1 = dominant strategy visualization (NEE=green, VC=blue, VM=red, BRDF=white, splat=yellow)
  - 2 = light vertex density heatmap at first camera hit (blue=0, green=moderate, red=many)
- [x] Per-strategy radiance accumulators (`radiance_nee`, `radiance_vc`, `radiance_vm`, `radiance_brdf`) tracked only when `debug_mode != 0`
- [x] Debug modes skip camera exposure to output fixed diagnostic colors
- [x] Light vertex density query reuses existing hash grid infrastructure (27-cell neighborhood)

#### 10.1 Reference Comparison (requires GPU)
- [ ] Render Cornell box scene with:
  - Pure path tracing (existing implementation) — baseline
  - Pure BPT (VC only, `enable_vm = false`)
  - Pure BPM (VM only, `enable_vc = false`)
  - Full VCM (both enabled)
- [ ] Compare convergence rates, especially for:
  - Caustics (light → specular → diffuse) — VM should excel
  - Direct illumination — NEE/VC should handle efficiently
  - Indirect diffuse — all methods comparable
  - SDS paths (specular-diffuse-specular) — VCM's key advantage

#### 10.2 Correctness Tests (requires GPU)
- [ ] Verify energy conservation: white furnace test (diffuse sphere in uniform environment)
- [ ] Verify MIS weight correctness: sum of weights across all strategies = 1.0 for any path
- [ ] Check for light leaks, dark spots, or fireflies indicating weight errors
- [ ] Validate progressive convergence: image should approach ground truth as iterations increase

#### 10.3 Performance Benchmarks (requires GPU)
- [ ] Measure per-frame time breakdown: light trace / hash build / camera trace
- [ ] Compare total time-to-convergence (not per-frame time) against unidirectional PT
- [ ] Profile GPU occupancy and memory bandwidth utilization
- [ ] Test at multiple resolutions and max path lengths

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU memory for light vertex storage | High — can exceed VRAM on complex scenes | Budget-capped storage with reservoir sampling |
| Hash grid prefix sum performance | Medium — serial bottleneck on GPU | Use well-optimized parallel scan (Blelloch or Hillis-Steele) |
| Ray coherence degradation from bidirectional tracing | Medium — reduced TLAS traversal efficiency | Wavefront architecture, path sorting |
| MIS weight numerical instability | High — incorrect weights cause fireflies or energy loss | Extensive validation, clamping, NaN guards |
| Merge radius tuning sensitivity | Medium — too large = blurry, too small = noisy | Progressive reduction, per-pixel adaptive radius |
| Atomic splatting for light tracing | Low-Medium — precision loss or contention | Separate splatting buffer with compositing pass |

---

## Suggested Implementation Order

1. **Phase 6** (BRDF extensions) — low risk, needed by everything else
2. **Phase 1** (Light subpath storage + tracing) — core new infrastructure
3. **Phase 2** (Hash grid) — needed for vertex merging
4. **Phase 3** (Camera path modifications) — the main integration point
5. **Phase 4** (MIS balancing) — tuning and correctness
6. **Phase 5** (Render graph restructuring) — wiring it all together
7. **Phase 7** (Camera connection / light tracing) — optional but improves quality
8. **Phase 8** (Accumulation updates) — adapt existing convergence system
9. **Phase 9** (Optimization) — after correctness is established
10. **Phase 10** (Testing) — ongoing throughout, formal validation at end

Start with BPT-only (vertex connections without merging) as an intermediate milestone. Once BPT is correct and MIS weights are validated, add the hash grid and vertex merging to complete VCM.
