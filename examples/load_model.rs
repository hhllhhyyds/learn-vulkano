use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;
use std::time::Instant;

use glam::Vec3Swizzles;

use vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Subpass;
use vulkano::swapchain::{self, AcquireError, SwapchainPresentInfo};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::{
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
};

use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use learn_vulkano::light::{AmbientLight, DirectionalLight};
use learn_vulkano::obj_loader::{DummyVertex, Model, NormalVertex};

use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "examples/load_model/shaders/deferred.frag"
    }
}

mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "examples/load_model/shaders/deferred.vert",
        types_meta: { #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)] },
    }
}

mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "examples/load_model/shaders/directional.frag",
        types_meta: { #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)] },
    }
}

mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "examples/load_model/shaders/directional.vert",
    }
}

mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "examples/load_model/shaders/ambient.vert",
    }
}

mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "examples/load_model/shaders/ambient.frag",
        types_meta: { #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)] },
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

    let render_pass = vulkano::ordered_passes_renderpass!(device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            color: {
                load: Clear,
                store: DontCare,
                format: Format::A2B10G10R10_UNORM_PACK32,
                samples: 1,
            },
            normals: {
                load: Clear,
                store: DontCare,
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        passes: [
            {
                color: [color, normals],
                depth_stencil: {depth},
                input: []
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: [color, normals]
            }
        ]
    )
    .unwrap();
    let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
    let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

    let deferred_vert = deferred_vert::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag::load(device.clone()).unwrap();
    let directional_vert = directional_vert::load(device.clone()).unwrap();
    let directional_frag = directional_frag::load(device.clone()).unwrap();
    let ambient_vert = ambient_vert::load(device.clone()).unwrap();
    let ambient_frag = ambient_frag::load(device.clone()).unwrap();

    let part_pipeline = GraphicsPipeline::start()
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back));

    let deferred_pipeline = part_pipeline
        .clone()
        .vertex_input_state(BuffersDefinition::new().vertex::<NormalVertex>())
        .vertex_shader(deferred_vert.entry_point("main").unwrap(), ())
        .fragment_shader(deferred_frag.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(deferred_pass)
        .build(device.clone())
        .unwrap();

    let ambient_pipeline = part_pipeline
        .clone()
        .vertex_input_state(BuffersDefinition::new().vertex::<DummyVertex>())
        .vertex_shader(ambient_vert.entry_point("main").unwrap(), ())
        .fragment_shader(ambient_frag.entry_point("main").unwrap(), ())
        .color_blend_state(
            ColorBlendState::new(lighting_pass.num_color_attachments()).blend(AttachmentBlend {
                color_op: BlendOp::Add,
                color_source: BlendFactor::One,
                color_destination: BlendFactor::One,
                alpha_op: BlendOp::Max,
                alpha_source: BlendFactor::One,
                alpha_destination: BlendFactor::One,
            }),
        )
        .render_pass(lighting_pass.clone())
        .build(device.clone())
        .unwrap();

    let directional_pipeline = part_pipeline
        .clone()
        .vertex_input_state(BuffersDefinition::new().vertex::<DummyVertex>())
        .vertex_shader(directional_vert.entry_point("main").unwrap(), ())
        .color_blend_state(
            ColorBlendState::new(lighting_pass.num_color_attachments()).blend(AttachmentBlend {
                color_op: BlendOp::Add,
                color_source: BlendFactor::One,
                color_destination: BlendFactor::One,
                alpha_op: BlendOp::Max,
                alpha_source: BlendFactor::One,
                alpha_destination: BlendFactor::One,
            }),
        )
        .fragment_shader(directional_frag.entry_point("main").unwrap(), ())
        .render_pass(lighting_pass.clone())
        .build(device.clone())
        .unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let mut model = Model::builder("models/suzanne.obj").build();
    model.translate(glam::vec3(0.0, 0.0, -3.0));

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        model.data().iter().cloned(),
    )
    .unwrap();

    let dummy_verts = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        DummyVertex::list().iter().cloned(),
    )
    .unwrap();

    let mut vp = learn_vulkano::mvp::VP::new();
    let vp_buffer = CpuAccessibleBuffer::from_data(
        &memory_allocator,
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        deferred_vert::ty::VpData {
            view: vp.view.to_cols_array_2d(),
            projection: vp.projection.to_cols_array_2d(),
        },
    )
    .unwrap();

    let ambient_light = AmbientLight {
        color: [1.0, 1.0, 1.0],
        intensity: 0.2,
    };
    let directional_light_w = DirectionalLight {
        position: [-4.0, 0.0, -4.0],
        color: [2.0, 2.0, 2.0],
    };

    let (mut framebuffers, mut color_buffer, mut normal_buffer) =
        create_framebuffer_and_other(&images, render_pass.clone(), &memory_allocator);

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let vp_layout = deferred_pipeline.layout().set_layouts().first().unwrap();
    let mut vp_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        vp_layout.clone(),
        [WriteDescriptorSet::buffer(0, vp_buffer)],
    )
    .unwrap();

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

            let model_uniform_buffer: CpuBufferPool<deferred_vert::ty::ModelData> =
                CpuBufferPool::uniform_buffer(memory_allocator.clone());
            let model_uniform_subbuffer = {
                let elapsed = rotation_start.elapsed().as_secs() as f64
                    + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                let elapsed_as_radians = elapsed * PI as f64 / 180.0;

                model.zero_rotation();
                model.rotate(PI, glam::vec3(0.0, 1.0, 0.0));
                model.rotate(elapsed_as_radians as f32 * 10.0, glam::vec3(1.0, 0.0, 0.0));
                model.rotate(elapsed_as_radians as f32 * 30.0, glam::vec3(0.0, 1.0, 0.0));
                model.rotate(elapsed_as_radians as f32 * 50.0, glam::vec3(0.0, 0.0, 1.0));

                let uniform_data = deferred_vert::ty::ModelData {
                    model: model.model_matrix().to_cols_array_2d(),
                    normals: model.normal_matrix().to_cols_array_2d(),
                };

                model_uniform_buffer.from_data(uniform_data).unwrap()
            };

            let ambient_buffer: CpuBufferPool<ambient_frag::ty::AmbientLightData> =
                CpuBufferPool::uniform_buffer(memory_allocator.clone());
            let ambient_subbuffer = {
                let uniform_data = ambient_frag::ty::AmbientLightData {
                    color: ambient_light.color,
                    intensity: ambient_light.intensity,
                };

                ambient_buffer.from_data(uniform_data).unwrap()
            };

            let directional_buffer: CpuBufferPool<directional_frag::ty::DirectionalLightData> =
                CpuBufferPool::uniform_buffer(memory_allocator.clone());

            let model_layout = deferred_pipeline.layout().set_layouts().get(1).unwrap();
            let model_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                model_layout.clone(),
                [WriteDescriptorSet::buffer(
                    0,
                    model_uniform_subbuffer.clone(),
                )],
            )
            .unwrap();

            let ambient_layout = ambient_pipeline.layout().set_layouts().first().unwrap();
            let ambient_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                ambient_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer.clone()),
                    WriteDescriptorSet::buffer(1, ambient_subbuffer),
                ],
            )
            .unwrap();

            if recreate_swapchain {
                let image_extent: [u32; 2] = learn_vulkano::setup::surface_extent(&surface).into();
                let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
                vp.projection = glam::Mat4::perspective_rh_gl(FRAC_PI_2, aspect_ratio, 0.01, 100.0);

                let (new_swapchain, new_images) = learn_vulkano::setup::create_swapchain_and_images(
                    device.clone(),
                    surface.clone(),
                    Some(swapchain.clone()),
                );
                let (new_framebuffers, new_color_buffer, new_normal_buffer) =
                    create_framebuffer_and_other(
                        &new_images,
                        render_pass.clone(),
                        &memory_allocator,
                    );

                swapchain = new_swapchain;
                framebuffers = new_framebuffers;
                color_buffer = new_color_buffer;
                normal_buffer = new_normal_buffer;

                let new_vp_buffer = CpuAccessibleBuffer::from_data(
                    &memory_allocator,
                    BufferUsage {
                        uniform_buffer: true,
                        ..BufferUsage::empty()
                    },
                    false,
                    deferred_vert::ty::VpData {
                        view: vp.view.to_cols_array_2d(),
                        projection: vp.projection.to_cols_array_2d(),
                    },
                )
                .unwrap();

                let vp_layout = deferred_pipeline.layout().set_layouts().first().unwrap();
                vp_set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    vp_layout.clone(),
                    [WriteDescriptorSet::buffer(0, new_vp_buffer)],
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

            let clear_values = vec![
                Some([0.0, 0.0, 0.0, 1.0].into()),
                Some([0.0, 0.0, 0.0, 1.0].into()),
                Some([0.0, 0.0, 0.0, 1.0].into()),
                Some(1.0.into()),
            ];

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
                .bind_pipeline_graphics(deferred_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    deferred_pipeline.layout().clone(),
                    0,
                    (vp_set.clone(), model_set.clone()),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .next_subpass(SubpassContents::Inline)
                .unwrap();

            let directional_layout = directional_pipeline.layout().set_layouts().first().unwrap();

            let directional_subbuffer =
                generate_directional_buffer(&directional_buffer, &directional_light_w);
            let directional_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                directional_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer.clone()),
                    WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                    WriteDescriptorSet::buffer(2, directional_subbuffer),
                ],
            )
            .unwrap();
            cmd_buffer_builder
                .bind_pipeline_graphics(directional_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    directional_pipeline.layout().clone(),
                    0,
                    directional_set.clone(),
                )
                .bind_vertex_buffers(0, dummy_verts.clone())
                .draw(dummy_verts.len() as u32, 1, 0, 0)
                .unwrap();

            cmd_buffer_builder
                .bind_pipeline_graphics(ambient_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    ambient_pipeline.layout().clone(),
                    0,
                    ambient_set.clone(),
                )
                .bind_vertex_buffers(0, dummy_verts.clone())
                .draw(dummy_verts.len() as u32, 1, 0, 0)
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

fn generate_directional_buffer(
    pool: &CpuBufferPool<directional_frag::ty::DirectionalLightData>,
    light: &learn_vulkano::light::DirectionalLight,
) -> Arc<CpuBufferPoolSubbuffer<directional_frag::ty::DirectionalLightData>> {
    let position = glam::Vec3::from_array(light.position);
    let uniform_data = directional_frag::ty::DirectionalLightData {
        position: position.xyzz().into(),
        color: light.color,
    };

    pool.from_data(uniform_data).unwrap()
}

#[allow(clippy::type_complexity)]
pub fn create_framebuffer_and_other(
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
