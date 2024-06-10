use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;
use std::time::Instant;

use glam::Vec3Swizzles;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::{CommandBufferUsage, RenderPassBeginInfo, SubpassContents};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use vulkano::{
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    render_pass::{Framebuffer, FramebufferCreateInfo, FramebufferCreationError, RenderPass},
};

use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use learn_vulkano::setup::RenderBase;

mod vertex;
use vertex::VertexB;

mod vs {
    use learn_vulkano::mvp::MVP;
    use ty::MvpData;

    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec3 color;

            layout(location = 0) out vec3 out_color;
            layout(location = 1) out vec3 out_normal;
            layout(location = 2) out vec3 frag_pos;

            layout(set = 0, binding = 0) uniform MvpData {
                mat4 model;
                mat4 view;
                mat4 projection;
            } uniforms;

            void main() {
                mat4 worldview = uniforms.view * uniforms.model;
                gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
                out_color = color;
                out_normal = mat3(uniforms.model) * normal;
                frag_pos = vec3(uniforms.model * vec4(position, 1.0));
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
            layout(location = 1) in vec3 in_normal;
            layout(location = 2) in vec3 frag_pos;
            
            layout(location = 0) out vec4 f_color;
            
            layout(set = 0, binding = 1) uniform AmbientLightData {
                vec3 color;
                float intensity;
            } ambient;

            layout(set = 0, binding = 2) uniform DirectionalLightData {
                vec4 position;
                vec3 color;
            } directional;
            
            void main() {
                vec3 ambient_color = ambient.intensity * ambient.color;
                vec3 light_direction = normalize(directional.position.xyz - frag_pos);
                float directional_intensity = max(dot(in_normal, light_direction), 0.0);
                vec3 directional_color = directional_intensity * directional.color;
                vec3 combined_color = (ambient_color + directional_color) * in_color;
                f_color = vec4(combined_color, 1.0);
            }
        ", types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
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
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let vs = vs::load(render_base.device()).unwrap();
    let fs = fs::load(render_base.device()).unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<VertexB>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(render_base.device())
        .unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(render_base.device()));
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(render_base.device());

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        vertex::CUBE_B,
    )
    .unwrap();

    let mut mvp = learn_vulkano::mvp::MVP::new();
    let ambient_light = learn_vulkano::light::AmbientLight {
        color: [1.0, 1.0, 1.0],
        intensity: 0.2,
    };
    let directional_light = learn_vulkano::light::DirectionalLight {
        position: [-4.0, -4.0, 0.0],
        color: [1.0, 1.0, 1.0],
    };

    let uniform_buffer: CpuBufferPool<vs::ty::MvpData> =
        CpuBufferPool::uniform_buffer(memory_allocator.clone());
    let ambient_buffer: CpuBufferPool<fs::ty::AmbientLightData> =
        CpuBufferPool::uniform_buffer(memory_allocator.clone());
    let directional_buffer: CpuBufferPool<fs::ty::DirectionalLightData> =
        CpuBufferPool::uniform_buffer(memory_allocator.clone());

    let ambient_subbuffer = {
        let uniform_data = fs::ty::AmbientLightData {
            color: ambient_light.color,
            intensity: ambient_light.intensity,
        };

        ambient_buffer.from_data(uniform_data).unwrap()
    };

    let directional_subbuffer = {
        let position = glam::Vec3::from_array(directional_light.position);
        let uniform_data = fs::ty::DirectionalLightData {
            position: position.xyzz().into(),
            color: directional_light.color,
        };

        directional_buffer.from_data(uniform_data).unwrap()
    };

    let mut framebuffers = create_framebuffers(
        &render_base.swapchain_images(),
        render_pass.clone(),
        &memory_allocator,
    )
    .unwrap();
    let recreate_swapchain =
        |render_base: &mut RenderBase,
         framebuffers: &mut Vec<Arc<Framebuffer>>,
         render_pass: Arc<RenderPass>,
         memory_allocator: Arc<StandardMemoryAllocator>| {
            render_base.recreate_swapchain();
            *framebuffers = create_framebuffers(
                &render_base.swapchain_images(),
                render_pass.clone(),
                &memory_allocator,
            )
            .unwrap();
        };

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
                recreate_swapchain(
                    &mut render_base,
                    &mut framebuffers,
                    render_pass.clone(),
                    memory_allocator.clone(),
                );
            }
            Event::RedrawEventsCleared => {
                render_base.frame_cleanup_finished();

                let uniform_subbuffer = {
                    let elapsed = rotation_start.elapsed().as_secs() as f64
                        + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                    let elapsed_as_radians = elapsed * PI as f64 / 180.0;

                    let image_extent: [u32; 2] = render_base.surface_extent().into();

                    let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
                    mvp.projection =
                        glam::Mat4::perspective_rh_gl(FRAC_PI_2, aspect_ratio, 0.01, 100.0);
                    mvp.model = glam::Mat4::from_translation((0.0, 0.0, -5.0).into())
                        * glam::Mat4::from_rotation_z(elapsed_as_radians as f32 * 50.0)
                        * glam::Mat4::from_rotation_y(elapsed_as_radians as f32 * 30.0)
                        * glam::Mat4::from_rotation_x(elapsed_as_radians as f32 * 20.0);

                    uniform_buffer.from_data(mvp.into()).unwrap()
                };

                let layout = pipeline.layout().set_layouts().first().unwrap();
                let set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, uniform_subbuffer),
                        WriteDescriptorSet::buffer(1, ambient_subbuffer.clone()),
                        WriteDescriptorSet::buffer(2, directional_subbuffer.clone()),
                    ],
                )
                .unwrap();

                let Some((image_index, acquire_future)) = render_base.acquire_next_image() else {
                    recreate_swapchain(
                        &mut render_base,
                        &mut framebuffers,
                        render_pass.clone(),
                        memory_allocator.clone(),
                    );
                    return;
                };

                let clear_values = vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())];

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
                    recreate_swapchain(
                        &mut render_base,
                        &mut framebuffers,
                        render_pass.clone(),
                        memory_allocator.clone(),
                    );
                }
            }
            _ => {}
        });
}

pub fn create_framebuffers(
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
