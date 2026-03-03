use super::{prepare::PathtracerAccumulationTexture, Pathtracer};
use bevy_camera::Camera;
use bevy_ecs::{
    change_detection::DetectChanges,
    system::{Commands, Query},
    world::Ref,
};
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;

/// Extraction system that copies [`Pathtracer`] data from the main world
/// into the render world each frame.
///
/// For every 3D camera that has a [`Pathtracer`] component **and** is active:
/// - Clones the [`Pathtracer`] component onto the corresponding render-world
///   entity.
/// - Automatically sets [`Pathtracer::reset`] to `true` if the camera's
///   [`GlobalTransform`] changed since the last frame, so that the
///   accumulation buffer is cleared whenever the camera moves.
///
/// If the camera is inactive or the [`Pathtracer`] component was removed,
/// both the [`Pathtracer`] and [`PathtracerAccumulationTexture`] components
/// are removed from the render-world entity to stop path tracing.
pub fn extract_pathtracer(
    cameras_3d: Extract<
        Query<(
            RenderEntity,
            &Camera,
            Ref<GlobalTransform>,
            Option<&Pathtracer>,
        )>,
    >,
    mut commands: Commands,
) {
    for (entity, camera, global_transform, pathtracer) in &cameras_3d {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(pathtracer) = pathtracer
            && camera.is_active
        {
            let mut pathtracer = pathtracer.clone();
            pathtracer.reset |= global_transform.is_changed();
            entity_commands.insert(pathtracer);
        } else {
            entity_commands.remove::<(Pathtracer, PathtracerAccumulationTexture)>();
        }
    }
}
