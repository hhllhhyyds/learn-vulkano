use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;
use std::time::Instant;

use glam::Vec3Swizzles;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use vulkano::swapchain::{self, AcquireError, SwapchainPresentInfo};
use vulkano::sync::{self, FlushError, GpuFuture};

use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use learn_vulkano::vertex::VertexB;

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

mod model {
    use super::VertexB;
    pub const CUBE: [VertexB; 36] = [
        // front face
        VertexB {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, 1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, 1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, -1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        // back face
        VertexB {
            position: [1.000000, -1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, 1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, -1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, -1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        // top face
        VertexB {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, -1.000000, 1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, -1.000000, -1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, -1.000000, -1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, -1.000000, -1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        // bottom face
        VertexB {
            position: [1.000000, 1.000000, 1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, 1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, 1.000000, 1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, 1.000000, -1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        // left face
        VertexB {
            position: [-1.000000, -1.000000, -1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, 1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, -1.000000, -1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, 1.000000, 1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        // right face
        VertexB {
            position: [1.000000, -1.000000, 1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, 1.000000, 1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, 1.000000, -1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, -1.000000, 1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, 1.000000, -1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        VertexB {
            position: [1.000000, -1.000000, -1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
    ];
}

fn main() {
    let instance = learn_vulkano::instance::instance_for_window_requirements().unwrap();

    let (event_loop, surface) =
        learn_vulkano::window::window_eventloop_surface(instance.clone()).unwrap();

    let daq = learn_vulkano::device::device_and_queue_for_window_requirements(
        instance.clone(),
        surface.clone(),
    )
    .unwrap();
    let (device, queue) = (daq.logical, daq.queues[0].clone());

    let (mut swapchain, images) = learn_vulkano::swapchain::create_swapchain_and_images(
        device.clone(),
        surface.clone(),
        None,
    )
    .unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
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

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<VertexB>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let vertices = model::CUBE;
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

    let mut mvp = learn_vulkano::mvp::MVP::new();
    let ambient_light = learn_vulkano::light::AmbientLight {
        color: [1.0, 1.0, 1.0],
        intensity: 0.2,
    };
    let directional_light = learn_vulkano::light::DirectionalLight {
        position: [-4.0, -4.0, 0.0],
        color: [1.0, 1.0, 1.0],
    };

    let mut framebuffers = learn_vulkano::swapchain::create_framebuffer_with_depth(
        &images,
        render_pass.clone(),
        &memory_allocator,
    )
    .unwrap();

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

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
                let elapsed = rotation_start.elapsed().as_secs() as f64
                    + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                let elapsed_as_radians = elapsed * PI as f64 / 180.0;

                let image_extent: [u32; 2] =
                    learn_vulkano::swapchain::surface_extent(&surface).into();

                let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
                mvp.projection =
                    glam::Mat4::perspective_rh_gl(FRAC_PI_2, aspect_ratio, 0.01, 100.0);
                mvp.model = glam::Mat4::from_translation((0.0, 0.0, -5.0).into())
                    * glam::Mat4::from_rotation_z(elapsed_as_radians as f32 * 50.0)
                    * glam::Mat4::from_rotation_y(elapsed_as_radians as f32 * 30.0)
                    * glam::Mat4::from_rotation_x(elapsed_as_radians as f32 * 20.0);

                uniform_buffer.from_data(mvp.into()).unwrap()
            };

            let ambient_buffer: CpuBufferPool<fs::ty::AmbientLightData> =
                CpuBufferPool::uniform_buffer(memory_allocator.clone());
            let ambient_subbuffer = {
                let uniform_data = fs::ty::AmbientLightData {
                    color: ambient_light.color,
                    intensity: ambient_light.intensity,
                };

                ambient_buffer.from_data(uniform_data).unwrap()
            };

            let directional_buffer: CpuBufferPool<fs::ty::DirectionalLightData> =
                CpuBufferPool::uniform_buffer(memory_allocator.clone());
            let directional_subbuffer = {
                let position = glam::Vec3::from_array(directional_light.position);
                let uniform_data = fs::ty::DirectionalLightData {
                    position: position.xyzz().into(),
                    color: directional_light.color,
                };

                directional_buffer.from_data(uniform_data).unwrap()
            };

            let layout = pipeline.layout().set_layouts().first().unwrap();
            let set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform_subbuffer),
                    WriteDescriptorSet::buffer(1, ambient_subbuffer),
                    WriteDescriptorSet::buffer(2, directional_subbuffer),
                ],
            )
            .unwrap();

            if recreate_swapchain {
                let (new_swapchain, new_images) =
                    learn_vulkano::swapchain::create_swapchain_and_images(
                        device.clone(),
                        surface.clone(),
                        Some(swapchain.clone()),
                    )
                    .unwrap();
                swapchain = new_swapchain;
                framebuffers = learn_vulkano::swapchain::create_framebuffer_with_depth(
                    &new_images,
                    render_pass.clone(),
                    &memory_allocator,
                )
                .unwrap();
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

            let clear_values = vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())];

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
                        dimensions: learn_vulkano::swapchain::surface_extent(&surface).into(),
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
