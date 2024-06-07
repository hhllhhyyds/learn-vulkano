use std::sync::Arc;

use vulkano::{
    device::Device,
    image::SwapchainImage,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
};

use winit::{dpi::PhysicalSize, window::Window};

pub fn surface_extent(surface: &Surface) -> PhysicalSize<u32> {
    surface
        .object()
        .expect("Failed to get surface extent")
        .downcast_ref::<Window>()
        .expect("Failed to get surface extent")
        .inner_size()
}

pub fn create_swapchain_and_images(
    device: Arc<Device>,
    surface: Arc<Surface>,
    old_swapchain: Option<Arc<Swapchain>>,
) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
    let image_extent: [u32; 2] = surface_extent(&surface).into();

    if let Some(old_swapchain) = old_swapchain {
        old_swapchain
            .recreate(SwapchainCreateInfo {
                image_extent,
                ..old_swapchain.create_info()
            })
            .expect("Failed to recreate swapchain")
    } else {
        let caps = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .expect("Failed to get surface capabilities of physical device");

        let usage = caps.supported_usage_flags;
        let alpha = caps
            .supported_composite_alpha
            .iter()
            .next()
            .expect("Failed to get supported composite alpha mode for swapchain");

        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .expect("Failed to get surface format")[0]
                .0,
        );

        Swapchain::new(
            device,
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent,
                image_usage: usage,
                composite_alpha: alpha,
                ..Default::default()
            },
        )
        .expect("Failed to create swapchain")
    }
}
