use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    memory::allocator::StandardMemoryAllocator,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
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

#[allow(clippy::type_complexity)]
pub fn create_framebuffer(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    allocator: &StandardMemoryAllocator,
) -> (
    Vec<Arc<Framebuffer>>,
    Arc<ImageView<AttachmentImage>>,
    Arc<ImageView<AttachmentImage>>,
) {
    let mut framebuffers = vec![];
    let dimensions = images[0].dimensions().width_height();

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(allocator, dimensions, Format::D16_UNORM)
            .expect("Failed to create depth image"),
    )
    .expect("Failed to create depth image view");

    let color_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            allocator,
            dimensions,
            Format::A2B10G10R10_UNORM_PACK32,
        )
        .expect("Failed to create color input image"),
    )
    .expect("Failed to create color input image view");

    let normal_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            allocator,
            dimensions,
            Format::R16G16B16A16_SFLOAT,
        )
        .expect("Failed to create normal input image"),
    )
    .expect("Failed to create normal input image view");

    for image in images {
        let view =
            ImageView::new_default(image.clone()).expect("Failed to create swapchain image view");
        framebuffers.push(
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        color_buffer.clone(),
                        normal_buffer.clone(),
                        depth_buffer.clone(),
                    ],
                    ..Default::default()
                },
            )
            .expect("Failed to create framebuffer"),
        );
    }
    (framebuffers, color_buffer.clone(), normal_buffer.clone())
}
