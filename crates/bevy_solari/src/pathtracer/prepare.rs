use super::Pathtracer;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::ToExtents;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{TextureDescriptor, TextureDimension, TextureFormat, TextureUsages},
    renderer::RenderDevice,
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

/// Prepares (allocates or re-uses) the accumulation texture for every camera
/// that has a [`Pathtracer`] component.
///
/// Runs in the [`RenderSystems::PrepareResources`] set. For each pathtracing
/// camera it creates a [`TextureDescriptor`] matching the camera's physical
/// viewport size and fetches a [`CachedTexture`] from the [`TextureCache`].
/// The resulting [`PathtracerAccumulationTexture`] component is inserted onto
/// the camera entity so it is available to the [`pathtracer`](super::node::pathtracer)
/// compute pass later in the frame.
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

        let descriptor = TextureDescriptor {
            label: Some("pathtracer_accumulation_texture"),
            size: viewport.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        commands
            .entity(entity)
            .insert(PathtracerAccumulationTexture(
                texture_cache.get(&render_device, descriptor),
            ));
    }
}
