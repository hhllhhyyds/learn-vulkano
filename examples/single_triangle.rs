use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Subpass;
use vulkano::{
    image::{view::ImageView, SwapchainImage},
    render_pass::{Framebuffer, FramebufferCreateInfo, FramebufferCreationError, RenderPass},
};

use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use learn_vulkano::setup::RenderBase;
use learn_vulkano::vertex::VertexA;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;

            layout(location = 0) out vec3 out_color;

            void main() {
                gl_Position = vec4(position, 1.0);
                out_color = color;
            }
        "
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
    let mut render_base = RenderBase::new();
    let render_pass = vulkano::single_pass_renderpass!(
        render_base.device(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: render_base.swapchain().image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let vs = vs::load(render_base.device()).unwrap();
    let fs = fs::load(render_base.device()).unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<VertexA>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(render_base.device())
        .unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(render_base.device()));
    let vertices = [
        VertexA {
            position: [-0.5, 0.5, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        VertexA {
            position: [0.5, 0.5, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        VertexA {
            position: [0.0, -0.5, 0.0],
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

    let mut framebuffers =
        create_framebuffers(&render_base.swapchain_images(), render_pass.clone()).unwrap();
    let recreate_swapchain = |render_base: &mut RenderBase,
                              framebuffers: &mut Vec<Arc<Framebuffer>>,
                              render_pass: Arc<RenderPass>| {
        render_base.recreate_swapchain();
        *framebuffers =
            create_framebuffers(&render_base.swapchain_images(), render_pass.clone()).unwrap();
    };

    render_base
        .eventloop_take()
        .run(move |event, _, control_flow| match event {
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
                recreate_swapchain(&mut render_base, &mut framebuffers, render_pass.clone());
            }
            Event::RedrawEventsCleared => {
                render_base.frame_cleanup_finished();

                let Some((image_index, acquire_future)) = render_base.acquire_next_image() else {
                    recreate_swapchain(&mut render_base, &mut framebuffers, render_pass.clone());
                    return;
                };

                let clear_values = vec![Some([0.0, 0.68, 1.0, 1.0].into())];

                let mut cmd_buffer_builder =
                    render_base.alloc_cmd_buf_builder(CommandBufferUsage::OneTimeSubmit);

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
                            dimensions: render_base.surface_extent().into(),
                            depth_range: 0.0..1.0,
                        }],
                    )
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = cmd_buffer_builder.build().unwrap();

                if render_base.execute_cmd_buffer(acquire_future, image_index, command_buffer) {
                    recreate_swapchain(&mut render_base, &mut framebuffers, render_pass.clone());
                }
            }
            _ => {}
        });
}

fn create_framebuffers(
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
