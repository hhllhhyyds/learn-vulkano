use std::sync::Arc;

use vulkano::{instance::Instance, swapchain::Surface};
use vulkano_win::{CreationError, VkSurfaceBuild};
use winit::{event_loop::EventLoop, window::WindowBuilder};

pub fn window_eventloop_surface(
    instance: Arc<Instance>,
) -> Result<(EventLoop<()>, Arc<Surface>), CreationError> {
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance)?;
    Ok((event_loop, surface))
}
