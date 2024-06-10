use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;
use std::time::Instant;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::{CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
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
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(render_base.device());

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

    let mut framebuffers =
        create_framebuffers(&render_base.swapchain_images(), render_pass.clone()).unwrap();
    let recreate_swapchain = |render_base: &mut RenderBase,
                              framebuffers: &mut Vec<Arc<Framebuffer>>,
                              render_pass: Arc<RenderPass>| {
        render_base.recreate_swapchain();
        *framebuffers =
            create_framebuffers(&render_base.swapchain_images(), render_pass.clone()).unwrap();
    };

    let uniform_buffer: CpuBufferPool<vs::ty::MvpData> =
        CpuBufferPool::uniform_buffer(memory_allocator.clone());

    let rotation_start = Instant::now();
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

                let uniform_subbuffer = {
                    let mut mvp = learn_vulkano::mvp::MVP::new();

                    let elapsed = rotation_start.elapsed().as_secs() as f64
                        + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                    let elapsed_as_radians = elapsed * PI as f64 / 180.0 * 30.0;

                    let image_extent: [u32; 2] = render_base.surface_extent().into();

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
                    .set_viewport(0, [render_base.full_viewport()])
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
