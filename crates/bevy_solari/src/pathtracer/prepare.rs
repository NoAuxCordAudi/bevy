use super::{Pathtracer, PathtracerSettingsUniform};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::ToExtents;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, UniformBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, TextureCache},
};

/// Component attached to a camera's render-world entity that holds the
/// accumulation texture for progressive path tracing.
///
/// The accumulation texture is an `Rgba32Float` storage texture sized to
/// match the camera viewport. The RGB channels store the running average of
/// all accumulated radiance samples, and the alpha channel stores the sample
/// count (used to compute the running average in the shader).
///
/// The texture is obtained from the [`TextureCache`] so it can be efficiently
/// reused across frames without repeated GPU allocations.
#[derive(Component)]
pub struct PathtracerAccumulationTexture(pub CachedTexture);

/// Component that holds the per-pixel variance (M2) texture for Welford's
/// online variance tracking. Format is `R32Float`, one float per pixel.
#[derive(Component)]
pub struct PathtracerVarianceTexture(pub CachedTexture);

/// Component that holds the GPU uniform buffer for convergence settings.
#[derive(Component)]
pub struct PathtracerSettingsBuffer(pub UniformBuffer<PathtracerSettingsUniform>);

/// Prepares (allocates or re-uses) the accumulation and variance textures for
/// every camera that has a [`Pathtracer`] component.
pub fn prepare_pathtracer_accumulation_texture(
    query: Query<(Entity, &ExtractedCamera), With<Pathtracer>>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, camera) in &query {
        let Some(viewport) = camera.physical_viewport_size else {
            continue;
        };

        let accumulation_descriptor = TextureDescriptor {
            label: Some("pathtracer_accumulation_texture"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let variance_descriptor = TextureDescriptor {
            label: Some("pathtracer_variance_texture"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        commands.entity(entity).insert((
            PathtracerAccumulationTexture(
                texture_cache.get(&render_device, accumulation_descriptor),
            ),
            PathtracerVarianceTexture(texture_cache.get(&render_device, variance_descriptor)),
        ));
    }
}

/// Creates and uploads the [`PathtracerSettingsBuffer`] uniform for each
/// pathtracing camera.
pub fn prepare_pathtracer_settings_buffer(
    query: Query<(Entity, &Pathtracer)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    for (entity, pathtracer) in &query {
        let mut buffer = UniformBuffer::from(PathtracerSettingsUniform {
            min_samples: pathtracer.min_samples,
            max_samples: pathtracer.max_samples,
            convergence_threshold: pathtracer.convergence_threshold,
        });
        buffer.write_buffer(&render_device, &render_queue);
        commands
            .entity(entity)
            .insert(PathtracerSettingsBuffer(buffer));
    }
}
