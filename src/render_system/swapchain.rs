use std::{f32::consts::FRAC_PI_2, sync::Arc};

use bytemuck::{Pod, Zeroable};

use glam::Mat4;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    memory::allocator::StandardMemoryAllocator,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
};

use winit::{dpi::PhysicalSize, window::Window};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VP {
    pub view: Mat4,
    pub projection: Mat4,
}

impl VP {
    pub fn new() -> VP {
        VP {
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
        }
    }

    pub fn from_surface(surface: &Surface) -> Self {
        let image_extent: [u32; 2] = surface_extent(surface).into();
        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        let projection = Mat4::perspective_rh_gl(FRAC_PI_2, aspect_ratio, 0.01, 100.0);
        Self {
            view: Mat4::IDENTITY,
            projection,
        }
    }

    pub fn create_vp_uniform_buffer(
        &self,
        memory_allocator: Arc<StandardMemoryAllocator>,
    ) -> Arc<CpuAccessibleBuffer<super::shaders::deferred_vert::ty::VpData>> {
        CpuAccessibleBuffer::from_data(
            &memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            super::shaders::deferred_vert::ty::VpData {
                view: self.view.to_cols_array_2d(),
                projection: self.projection.to_cols_array_2d(),
            },
        )
        .unwrap()
    }
}

impl Default for VP {
    fn default() -> Self {
        Self::new()
    }
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

pub fn surface_extent(surface: &Surface) -> PhysicalSize<u32> {
    surface
        .object()
        .unwrap()
        .downcast_ref::<Window>()
        .unwrap()
        .inner_size()
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
        AttachmentImage::transient(allocator, dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    let color_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            allocator,
            dimensions,
            Format::A2B10G10R10_UNORM_PACK32,
        )
        .unwrap(),
    )
    .unwrap();

    let normal_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            allocator,
            dimensions,
            Format::R16G16B16A16_SFLOAT,
        )
        .unwrap(),
    )
    .unwrap();

    for image in images {
        let view = ImageView::new_default(image.clone()).unwrap();
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
            .unwrap(),
        );
    }
    (framebuffers, color_buffer.clone(), normal_buffer.clone())
}
