use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
    },
    device::{Device, Queue},
    image::SwapchainImage,
    instance::Instance,
    pipeline::graphics::viewport::Viewport,
    swapchain::{AcquireError, Surface, Swapchain, SwapchainAcquireFuture, SwapchainPresentInfo},
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::WindowBuilder};

pub struct RenderBase {
    instance: Arc<Instance>,
    eventloop: Option<EventLoop<()>>,
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<SwapchainImage>>,
    std_cmd_buf_allocator: StandardCommandBufferAllocator,
    frame_end_future: Option<Box<dyn GpuFuture>>,
}

impl RenderBase {
    pub fn new() -> Self {
        let instance = super::instance::create_instance_for_window_app();
        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();
        let (device, queue) =
            super::device::DeviceAndQueue::new_for_window_app(instance.clone(), surface.clone())
                .get_device_and_first_queue();
        let (swapchain, swapchain_images) =
            super::swapchain::create_swapchain_and_images(device.clone(), surface.clone(), None);
        let std_cmd_buf_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let frame_end_future = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
        Self {
            instance,
            eventloop: Some(event_loop),
            surface,
            device,
            queue,
            swapchain,
            swapchain_images,
            std_cmd_buf_allocator,
            frame_end_future,
        }
    }

    pub fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    pub fn surface(&self) -> Arc<Surface> {
        self.surface.clone()
    }

    pub fn surface_extent(&self) -> PhysicalSize<u32> {
        super::surface_extent(&self.surface)
    }

    pub fn full_viewport(&self) -> Viewport {
        Viewport {
            origin: [0.0, 0.0],
            dimensions: self.surface_extent().into(),
            depth_range: 0.0..1.0,
        }
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    pub fn swapchain(&self) -> Arc<Swapchain> {
        self.swapchain.clone()
    }

    pub fn swapchain_images(&self) -> Vec<Arc<SwapchainImage>> {
        self.swapchain_images.clone()
    }

    pub fn eventloop_take(&mut self) -> EventLoop<()> {
        self.eventloop.take().expect("event loop already taken")
    }

    pub fn frame_end_future_set_now(&mut self) {
        self.frame_end_future = Some(Box::new(sync::now(self.device())) as Box<dyn GpuFuture>);
    }

    pub fn recreate_swapchain(&mut self) {
        let (new_swapchain, new_images) = super::swapchain::create_swapchain_and_images(
            self.device(),
            self.surface(),
            Some(self.swapchain.clone()),
        );
        self.swapchain = new_swapchain;
        self.swapchain_images = new_images;
    }

    pub fn acquire_next_image(&mut self) -> Option<(u32, SwapchainAcquireFuture)> {
        match vulkano::swapchain::acquire_next_image(self.swapchain(), None) {
            Ok((image_index, suboptimal, acquire_future)) => {
                if suboptimal {
                    self.recreate_swapchain();
                    None
                } else {
                    Some((image_index, acquire_future))
                }
            }
            Err(AcquireError::OutOfDate) => {
                self.recreate_swapchain();
                None
            }
            Err(e) => panic!("Failed to acquire next image: {:?}", e),
        }
    }

    pub fn alloc_cmd_buf_builder(
        &self,
        usage: CommandBufferUsage,
    ) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            &self.std_cmd_buf_allocator,
            self.queue.queue_family_index(),
            usage,
        )
        .expect("Failed to alloc command buffer builder")
    }

    pub fn frame_cleanup_finished(&mut self) {
        self.frame_end_future
            .as_mut()
            .take()
            .unwrap()
            .cleanup_finished()
    }

    pub fn execute_cmd_buffer(
        &mut self,
        swapchain_future: SwapchainAcquireFuture,
        image_index: u32,
        command_buffer: PrimaryAutoCommandBuffer,
    ) -> bool {
        let mut swapchain_need_recreation = false;

        let future = self
            .frame_end_future
            .take()
            .unwrap()
            .join(swapchain_future)
            .then_execute(self.queue(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain(), image_index),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.frame_end_future = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                swapchain_need_recreation = true;
                self.frame_end_future_set_now();
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.frame_end_future_set_now();
            }
        }

        swapchain_need_recreation
    }
}

impl Default for RenderBase {
    fn default() -> Self {
        Self::new()
    }
}
