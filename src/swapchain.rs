use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    memory::allocator::StandardMemoryAllocator,
    render_pass::{Framebuffer, FramebufferCreateInfo, FramebufferCreationError, RenderPass},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError},
};
use winit::{dpi::PhysicalSize, window::Window};

// TODO: clear unwraps

pub fn create_swapchain_and_images(
    device: Arc<Device>,
    surface: Arc<Surface>,
    old_swapchain: Option<Arc<Swapchain>>,
) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), SwapchainCreationError> {
    let image_extent: [u32; 2] = surface_extent(&surface).into();

    if let Some(old_swapchain) = old_swapchain {
        old_swapchain.recreate(SwapchainCreateInfo {
            image_extent,
            ..old_swapchain.create_info()
        })
    } else {
        let caps = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
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
    }
}

pub fn create_framebuffer(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
) -> Result<Vec<Arc<Framebuffer>>, FramebufferCreationError> {
    let mut framebuffer = vec![];
    for image in images {
        let view = ImageView::new_default(image.clone()).unwrap();
        framebuffer.push(Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        )?);
    }
    Ok(framebuffer)
}

pub fn create_framebuffer_with_depth(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    allocator: &StandardMemoryAllocator,
) -> Result<Vec<Arc<Framebuffer>>, FramebufferCreationError> {
    let mut framebuffer = vec![];
    let dimensions = images[0].dimensions().width_height();

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(allocator, dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    for image in images {
        let view = ImageView::new_default(image.clone()).unwrap();
        framebuffer.push(Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![view, depth_buffer.clone()],
                ..Default::default()
            },
        )?);
    }
    Ok(framebuffer)
}

pub fn surface_extent(surface: &Surface) -> PhysicalSize<u32> {
    surface
        .object()
        .unwrap()
        .downcast_ref::<Window>()
        .unwrap()
        .inner_size()
}
