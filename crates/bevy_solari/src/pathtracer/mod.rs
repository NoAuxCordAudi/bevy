mod extract;
mod node;
mod prepare;

use crate::SolariPlugins;
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_camera::Hdr;
use bevy_core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy_ecs::{component::Component, reflect::ReflectComponent, schedule::IntoScheduleConfigs};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    render_resource::ShaderType, renderer::RenderDevice, ExtractSchedule, Render, RenderApp,
    RenderStartup, RenderSystems,
};
use extract::extract_pathtracer;
use node::{init_pathtracer_pipelines, pathtracer};
use prepare::{prepare_pathtracer_accumulation_texture, prepare_pathtracer_settings_buffer};
use tracing::warn;

/// Non-realtime pathtracing.
///
/// This plugin is meant to generate reference screenshots to compare against,
/// and is not intended to be used by games.
pub struct PathtracingPlugin;

impl Plugin for PathtracingPlugin {
    /// Embeds the `pathtracer.wgsl` shader as an asset so it can be loaded
    /// at runtime without requiring an external file on disk.
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "pathtracer.wgsl");
    }

    /// Registers the pathtracer's render systems on the [`RenderApp`].
    ///
    /// Before registering anything, this checks that the GPU supports the
    /// required ray-tracing features (e.g. hardware ray queries). If the
    /// features are missing, a warning is logged and the plugin is skipped.
    ///
    /// The following systems are registered:
    /// - [`init_pathtracer_pipelines`] – creates the compute pipeline (runs once at startup).
    /// - [`extract_pathtracer`] – copies [`Pathtracer`] component data from the main world to the render world each frame.
    /// - [`prepare_pathtracer_accumulation_texture`] – allocates / resizes the accumulation texture to match the viewport.
    /// - [`pathtracer`] – dispatches the GPU compute pass that performs the actual path tracing, running after the main pass.
    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "PathtracingPlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        render_app
            .add_systems(RenderStartup, init_pathtracer_pipelines)
            .add_systems(ExtractSchedule, extract_pathtracer)
            .add_systems(
                Render,
                (
                    prepare_pathtracer_accumulation_texture,
                    prepare_pathtracer_settings_buffer,
                )
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(Core3d, pathtracer.after(Core3dSystems::MainPass));
    }
}

/// Component that enables path tracing for the camera it is attached to.
///
/// When present on a camera entity, the pathtracer will progressively
/// accumulate samples each frame and write the result into the camera's
/// view target. The accumulation buffer is preserved across frames so the
/// image converges over time.
///
/// Requires the [`Hdr`] component (added automatically via `#[require]`)
/// because the pathtracer outputs high-dynamic-range radiance values.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(Hdr)]
pub struct Pathtracer {
    /// When `true`, the accumulation buffer is cleared before the next frame,
    /// discarding all previously accumulated samples. This is set
    /// automatically when the camera's [`GlobalTransform`] changes (see
    /// [`extract_pathtracer`]), but can also be set manually to force a reset
    /// (e.g. after a scene change).
    pub reset: bool,
    /// Minimum number of samples per pixel before convergence testing begins.
    pub min_samples: u32,
    /// Maximum number of samples per pixel. Pixels stop accumulating after
    /// reaching this count regardless of convergence.
    pub max_samples: u32,
    /// Relative standard error threshold for per-pixel convergence. A pixel
    /// is considered converged when `sqrt(variance / n) / mean < threshold`.
    /// Set to `0.0` to disable convergence (all pixels run to `max_samples`).
    pub convergence_threshold: f32,
}

impl Default for Pathtracer {
    fn default() -> Self {
        Self {
            reset: false,
            min_samples: 64,
            max_samples: 4096,
            convergence_threshold: 0.01,
        }
    }
}

/// GPU-side uniform buffer matching the convergence settings in [`Pathtracer`].
#[derive(ShaderType)]
pub(crate) struct PathtracerSettingsUniform {
    pub min_samples: u32,
    pub max_samples: u32,
    pub convergence_threshold: f32,
}
