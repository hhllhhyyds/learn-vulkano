use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;
use std::time::Instant;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use vulkano::swapchain::{self, AcquireError, SwapchainPresentInfo};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::{
    image::{view::ImageView, SwapchainImage},
    render_pass::{Framebuffer, FramebufferCreateInfo, FramebufferCreationError, RenderPass},
};

use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use learn_vulkano::vertex::VertexA;

use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

mod vs {
    use learn_vulkano::mvp::MVP;
    use ty::MvpData;

    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;

            layout(location = 0) out vec3 out_color;

            layout(set = 0, binding = 0) uniform MvpData {
                mat4 model;
                mat4 view;
                mat4 projection;
            } uniforms;

            void main() {
                mat4 worldview = uniforms.view * uniforms.model;
                gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
                out_color = color;
            }
        ", types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }

    impl From<MVP> for MvpData {
        fn from(mvp: MVP) -> Self {
            ty::MvpData {
                model: mvp.model.to_cols_array_2d(),
                view: mvp.view.to_cols_array_2d(),
                projection: mvp.projection.to_cols_array_2d(),
            }
        }
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) in vec3 in_color;

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(in_color, 1.0);
            }
        "
    }
}

fn main() {
    let instance = learn_vulkano::setup::create_instance_for_window_app();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let (device, queue) =
        learn_vulkano::setup::DeviceAndQueue::new_for_window_app(instance.clone(), surface.clone())
            .get_device_and_first_queue();

    let (mut swapchain, images) =
        learn_vulkano::setup::create_swapchain_and_images(device.clone(), surface.clone(), None);

    // render pass descript shape of used data used in this pass
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<VertexA>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let vertices = [
        VertexA {
            position: [-0.5, 0.5, -1.0],
            color: [1.0, 0.0, 0.0],
        },
        VertexA {
            position: [0.5, 0.5, -1.0],
            color: [0.0, 1.0, 0.0],
        },
        VertexA {
            position: [0.0, -0.5, -1.0],
            color: [0.0, 0.0, 1.0],
        },
    ];

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        vertices,
    )
    .unwrap();

    let mut framebuffers = create_framebuffer(&images, render_pass.clone()).unwrap();

    // command buffer
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let rotation_start = Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            let uniform_buffer: CpuBufferPool<vs::ty::MvpData> =
                CpuBufferPool::uniform_buffer(memory_allocator.clone());

            let uniform_subbuffer = {
                let mut mvp = learn_vulkano::mvp::MVP::new();

                let elapsed = rotation_start.elapsed().as_secs() as f64
                    + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                let elapsed_as_radians = elapsed * PI as f64 / 180.0 * 30.0;

                let image_extent: [u32; 2] = learn_vulkano::setup::surface_extent(&surface).into();

                let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
                mvp.projection =
                    glam::Mat4::perspective_rh_gl(FRAC_PI_2, aspect_ratio, 0.01, 100.0);
                mvp.model = glam::Mat4::from_rotation_z(elapsed_as_radians as f32);

                uniform_buffer.from_data(mvp.into()).unwrap()
            };

            let layout = pipeline.layout().set_layouts().first().unwrap();
            let set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout.clone(),
                [WriteDescriptorSet::buffer(0, uniform_subbuffer)],
            )
            .unwrap();

            if recreate_swapchain {
                let (new_swapchain, new_images) = learn_vulkano::setup::create_swapchain_and_images(
                    device.clone(),
                    surface.clone(),
                    Some(swapchain.clone()),
                );
                swapchain = new_swapchain;
                framebuffers = create_framebuffer(&new_images, render_pass.clone()).unwrap();
                recreate_swapchain = false;
            }

            let (image_index, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let clear_values = vec![Some([0.0, 0.68, 1.0, 1.0].into())];

            let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            cmd_buffer_builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values,
                        ..RenderPassBeginInfo::framebuffer(
                            framebuffers[image_index as usize].clone(),
                        )
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .set_viewport(
                    0,
                    [Viewport {
                        origin: [0.0, 0.0],
                        dimensions: learn_vulkano::setup::surface_extent(&surface).into(),
                        depth_range: 0.0..1.0,
                    }],
                )
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set.clone(),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            let command_buffer = cmd_buffer_builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(Box::new(future) as Box<_>);
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                }
            }
        }
        _ => {}
    });
}

fn create_framebuffer(
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
