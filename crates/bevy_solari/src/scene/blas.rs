use bevy_asset::AssetId;
use tracing::warn;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
};
use bevy_mesh::{Indices, Mesh};
use bevy_platform::collections::HashMap;
use bevy_render::{
    mesh::{
        allocator::{MeshAllocator, MeshBufferSlice},
        RenderMesh,
    },
    render_asset::ExtractedAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, Blas>,
}

impl BlasManager {
    pub fn get(&self, mesh: &AssetId<Mesh>) -> Option<&Blas> {
        self.blas.get(mesh)
    }
}

pub fn generate_raytracing_tangents(
    mut extracted_meshes: ResMut<ExtractedAssets<RenderMesh>>,
) {
    // Auto-generate tangents for raytracing meshes that are missing them
    for (_, mesh) in extracted_meshes.extracted.iter_mut() {
        if mesh.enable_raytracing
            && mesh.contains_attribute(Mesh::ATTRIBUTE_POSITION)
            && mesh.contains_attribute(Mesh::ATTRIBUTE_NORMAL)
            && mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0)
            && !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT)
        {
            tracing::debug!(
                "Auto-generating tangents for raytracing mesh missing Vertex_Tangent attribute"
            );
            if let Err(err) = mesh.generate_tangents() {
                warn!(
                    "Failed to auto-generate tangents for raytracing: {}. Mesh will be skipped.",
                    err
                );
            }
        }
    }
}

pub fn prepare_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    extracted_meshes: ResMut<ExtractedAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    // Delete BLAS for deleted or modified meshes
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        blas_manager.blas.remove(asset_id);
    }

    if extracted_meshes.extracted.is_empty() {
        return;
    }

    // Create new BLAS for added or changed meshes
    let blas_resources = extracted_meshes
        .extracted
        .iter()
        .filter(|(_, mesh)| is_mesh_raytracing_compatible(mesh))
        .map(|(asset_id, _)| {
            let vertex_slice = mesh_allocator.mesh_vertex_slice(asset_id).unwrap();
            let index_slice = mesh_allocator.mesh_index_slice(asset_id).unwrap();

            let (blas, blas_size) =
                allocate_blas(&vertex_slice, &index_slice, asset_id, &render_device);

            blas_manager.blas.insert(*asset_id, blas);

            (*asset_id, vertex_slice, index_slice, blas_size)
        })
        .collect::<Vec<_>>();

    // Build geometry into each BLAS
    let build_entries = blas_resources
        .iter()
        .map(|(asset_id, vertex_slice, index_slice, blas_size)| {
            let geometry = BlasTriangleGeometry {
                size: blas_size,
                vertex_buffer: vertex_slice.buffer,
                first_vertex: vertex_slice.range.start,
                vertex_stride: 48,
                index_buffer: Some(index_slice.buffer),
                first_index: Some(index_slice.range.start),
                transform_buffer: None,
                transform_buffer_offset: None,
            };
            BlasBuildEntry {
                blas: &blas_manager.blas[asset_id],
                geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
            }
        })
        .collect::<Vec<_>>();

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_blas_command_encoder"),
    });
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    render_queue.submit([command_encoder.finish()]);
}

fn allocate_blas(
    vertex_slice: &MeshBufferSlice,
    index_slice: &MeshBufferSlice,
    asset_id: &AssetId<Mesh>,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count: vertex_slice.range.len() as u32,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_slice.range.len() as u32),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some(&asset_id.to_string()),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    (blas, blas_size)
}

fn is_mesh_raytracing_compatible(mesh: &Mesh) -> bool {
    let triangle_list = mesh.primitive_topology() == PrimitiveTopology::TriangleList;
    let expected_attrs = [
        Mesh::ATTRIBUTE_POSITION.id,
        Mesh::ATTRIBUTE_NORMAL.id,
        Mesh::ATTRIBUTE_UV_0.id,
        Mesh::ATTRIBUTE_TANGENT.id,
    ];
    let vertex_attributes = mesh.attributes().map(|(attribute, _)| attribute.id).eq(expected_attrs);
    let indexed_32 = matches!(mesh.indices(), Some(Indices::U32(..)));

    if mesh.enable_raytracing && triangle_list && !vertex_attributes && indexed_32 {
        let has_required = expected_attrs.iter().all(|req| {
            mesh.attributes().any(|(attr, _)| attr.id == *req)
        });
        let actual: Vec<_> = mesh.attributes().map(|(attr, _)| attr.name).collect();
        if has_required {
            warn!(
                "Mesh has required raytracing attributes but also has extras {:?}. \
                 Strip extra attributes for Solari compatibility.",
                actual
            );
        } else {
            warn!(
                "Mesh is missing required raytracing attributes. Has: {:?}, needs: \
                 [Position, Normal, UV_0, Tangent].",
                actual
            );
        }
    }

    mesh.enable_raytracing && triangle_list && vertex_attributes && indexed_32
}
