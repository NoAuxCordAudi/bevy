use super::{
    prepare::{PathtracerAccumulationTexture, PathtracerSettingsBuffer, PathtracerVarianceTexture},
    Pathtracer, PathtracerSettingsUniform,
};
use crate::scene::RaytracingSceneBindings;
use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_ecs::{prelude::*, resource::Resource, system::Commands};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        binding_types::{texture_storage_2d, uniform_buffer},
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        ImageSubresourceRange, PipelineCache, ShaderStages, StorageTextureAccess, TextureFormat,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_utils::default;

/// Resource that stores the GPU compute pipeline and bind-group layout used by
/// the pathtracer.
///
/// Created once during render startup by [`init_pathtracer_pipelines`] and
/// consumed each frame by the [`pathtracer`] system when dispatching the
/// compute pass.
#[derive(Resource)]
pub struct PathtracerPipelines {
    /// Layout describing the pathtracer-specific bindings (accumulation
    /// texture, view output texture, and view uniform buffer).
    bind_group_layout: BindGroupLayoutDescriptor,
    /// Handle to the cached compute pipeline compiled from `pathtracer.wgsl`.
    pipeline: CachedComputePipelineId,
}

/// One-shot startup system that creates the pathtracer compute pipeline and
/// its bind-group layout, then inserts them as the [`PathtracerPipelines`]
/// resource.
///
/// The bind-group layout contains three entries (all in `COMPUTE` stage):
/// 1. **Accumulation texture** (`Rgba32Float`, read-write) – running average
///    of all samples accumulated so far.
/// 2. **View output texture** (HDR, write-only) – the final image written to
///    the camera's view target for presentation.
/// 3. **View uniform buffer** – camera matrices, viewport size, exposure, etc.
///
/// The compute pipeline references two bind groups: bind group 0 is the
/// shared [`RaytracingSceneBindings`] (TLAS, meshes, materials, lights) and
/// bind group 1 is the layout defined here.
pub fn init_pathtracer_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    asset_server: Res<AssetServer>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "pathtracer_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(
                    ViewTarget::TEXTURE_FORMAT_HDR,
                    StorageTextureAccess::WriteOnly,
                ),
                uniform_buffer::<ViewUniform>(true),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                uniform_buffer::<PathtracerSettingsUniform>(false),
            ),
        ),
    );

    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("pathtracer_pipeline".into()),
        layout: vec![
            scene_bindings.bind_group_layout.clone(),
            bind_group_layout.clone(),
        ],
        shader: load_embedded_asset!(asset_server.as_ref(), "pathtracer.wgsl"),
        ..default()
    });

    commands.insert_resource(PathtracerPipelines {
        bind_group_layout,
        pipeline,
    });
}

/// Per-frame render system that dispatches the pathtracer compute shader.
///
/// This system runs after the main render pass in the [`Core3d`] schedule.
/// For each camera that has a [`Pathtracer`] component it:
///
/// 1. Creates a bind group with the accumulation texture, view output, and
///    view uniforms.
/// 2. If [`Pathtracer::reset`] is `true`, clears the accumulation texture so
///    sampling starts fresh.
/// 3. Begins a compute pass, binds the scene and pathtracer bind groups, and
///    dispatches enough 8×8 workgroups to cover the entire viewport.
///
/// The system exits early (no-op) if:
/// - The [`PathtracerPipelines`] resource is missing (GPU features unsupported).
/// - The compute pipeline has not finished compiling yet.
/// - The scene bind group or view uniforms are not yet available.
pub fn pathtracer(
    view: ViewQuery<(
        &Pathtracer,
        &PathtracerAccumulationTexture,
        &PathtracerVarianceTexture,
        &PathtracerSettingsBuffer,
        &ExtractedCamera,
        &ViewTarget,
        &ViewUniformOffset,
    )>,
    pathtracer_pipelines: Option<Res<PathtracerPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    scene_bindings: Res<RaytracingSceneBindings>,
    view_uniforms: Res<ViewUniforms>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (
        pathtracer_settings,
        accumulation_texture,
        variance_texture,
        settings_buffer,
        camera,
        view_target,
        view_uniform_offset,
    ) = view.into_inner();

    let Some(pathtracer_pipelines) = pathtracer_pipelines else {
        return;
    };

    let (
        Some(pipeline),
        Some(scene_bind_group),
        Some(viewport),
        Some(view_uniforms_binding),
        Some(settings_binding),
    ) = (
        pipeline_cache.get_compute_pipeline(pathtracer_pipelines.pipeline),
        &scene_bindings.bind_group,
        camera.physical_viewport_size,
        view_uniforms.uniforms.binding(),
        settings_buffer.0.binding(),
    ) else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "pathtracer_bind_group",
        &pipeline_cache.get_bind_group_layout(&pathtracer_pipelines.bind_group_layout),
        &BindGroupEntries::sequential((
            &accumulation_texture.0.default_view,
            view_target.get_unsampled_color_attachment().view,
            view_uniforms_binding,
            &variance_texture.0.default_view,
            settings_binding,
        )),
    );

    let command_encoder = ctx.command_encoder();

    if pathtracer_settings.reset {
        command_encoder.clear_texture(
            &accumulation_texture.0.texture,
            &ImageSubresourceRange::default(),
        );
        command_encoder.clear_texture(
            &variance_texture.0.texture,
            &ImageSubresourceRange::default(),
        );
    }

    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("pathtracer"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_bind_group(1, &bind_group, &[view_uniform_offset.offset]);
    pass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
}
