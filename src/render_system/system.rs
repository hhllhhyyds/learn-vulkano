#![allow(clippy::get_first)]

use std::sync::Arc;

use vulkano::{
    buffer::{
        cpu_pool::CpuBufferPoolSubbuffer, BufferUsage, CpuAccessibleBuffer, CpuBufferPool,
        TypedBufferAccess,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    instance::Instance,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, BlendFactor, BlendOp, ColorBlendState},
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{AcquireError, Surface, Swapchain, SwapchainAcquireFuture, SwapchainPresentInfo},
    sync::{FlushError, GpuFuture},
};

use vulkano_win::VkSurfaceBuild;

use winit::{event_loop::EventLoop, window::WindowBuilder};

use super::shaders;
use crate::{light, obj_loader};

#[derive(Debug, Clone)]
enum RenderStage {
    Stopped,
    Deferred,
    Ambient,
    Directional,
    LightObject,
    #[allow(unused)]
    NeedsRedraw,
}

pub struct RenderSystem {
    #[allow(unused)]
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    #[allow(unused)]
    swapchain_images: Vec<Arc<SwapchainImage>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,
    model_uniform_buffer_pool: CpuBufferPool<shaders::deferred_vert::ty::ModelData>,
    directional_uniform_buffer_pool:
        CpuBufferPool<shaders::directional_frag::ty::DirectionalLightData>,
    render_pass: Arc<RenderPass>,
    deferred_pipeline: Arc<GraphicsPipeline>,
    directional_pipeline: Arc<GraphicsPipeline>,
    ambient_pipeline: Arc<GraphicsPipeline>,
    light_obj_pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
    color_buffer: Arc<ImageView<AttachmentImage>>,
    normal_buffer: Arc<ImageView<AttachmentImage>>,
    dummy_verts: Arc<CpuAccessibleBuffer<[obj_loader::DummyVertex]>>,
    ambient_buffer: Arc<CpuAccessibleBuffer<shaders::ambient_frag::ty::AmbientLightData>>,
    vp: crate::mvp::VP,
    vp_buffer: Arc<CpuAccessibleBuffer<shaders::deferred_vert::ty::VpData>>,
    vp_set: Arc<PersistentDescriptorSet>,
    render_stage: RenderStage,
    commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    image_index: u32,
    acquire_future: Option<SwapchainAcquireFuture>,
}

impl RenderSystem {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let instance = crate::setup::create_instance_for_window_app();
        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
            .expect("Failed to build vulkan surface");
        let (device, queue) =
            crate::setup::DeviceAndQueue::new_for_window_app(instance.clone(), surface.clone())
                .get_device_and_first_queue();
        let (swapchain, swapchain_images) =
            crate::setup::create_swapchain_and_images(device.clone(), surface.clone(), None);
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let deferred_vert = shaders::deferred_vert::load(device.clone()).unwrap();
        let deferred_frag = shaders::deferred_frag::load(device.clone()).unwrap();
        let directional_vert = shaders::directional_vert::load(device.clone()).unwrap();
        let directional_frag = shaders::directional_frag::load(device.clone()).unwrap();
        let ambient_vert = shaders::ambient_vert::load(device.clone()).unwrap();
        let ambient_frag = shaders::ambient_frag::load(device.clone()).unwrap();
        let light_obj_vert = shaders::light_obj_vert::load(device.clone()).unwrap();
        let light_obj_frag = shaders::light_obj_frag::load(device.clone()).unwrap();

        let model_uniform_buffer_pool: CpuBufferPool<shaders::deferred_vert::ty::ModelData> =
            CpuBufferPool::uniform_buffer(memory_allocator.clone());
        let directional_uniform_buffer_pool: CpuBufferPool<
            shaders::directional_frag::ty::DirectionalLightData,
        > = CpuBufferPool::uniform_buffer(memory_allocator.clone());

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
                    depth_stencil: {depth},
                    input: [color, normals]
                }
            ]
        )
        .expect("Failed to create renderpass");

        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        let deferred_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<obj_loader::NormalVertex>())
            .vertex_shader(deferred_vert.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(deferred_frag.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .render_pass(deferred_pass.clone())
            .build(device.clone())
            .expect("Failed to create pipeline");

        let directional_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<obj_loader::DummyVertex>())
            .vertex_shader(directional_vert.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(directional_frag.entry_point("main").unwrap(), ())
            .color_blend_state(
                ColorBlendState::new(lighting_pass.num_color_attachments()).blend(
                    AttachmentBlend {
                        color_op: BlendOp::Add,
                        color_source: BlendFactor::One,
                        color_destination: BlendFactor::One,
                        alpha_op: BlendOp::Max,
                        alpha_source: BlendFactor::One,
                        alpha_destination: BlendFactor::One,
                    },
                ),
            )
            .render_pass(lighting_pass.clone())
            .build(device.clone())
            .expect("Failed to create pipeline");

        let ambient_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<obj_loader::DummyVertex>())
            .vertex_shader(ambient_vert.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(ambient_frag.entry_point("main").unwrap(), ())
            .color_blend_state(
                ColorBlendState::new(lighting_pass.num_color_attachments()).blend(
                    AttachmentBlend {
                        color_op: BlendOp::Add,
                        color_source: BlendFactor::One,
                        color_destination: BlendFactor::One,
                        alpha_op: BlendOp::Max,
                        alpha_source: BlendFactor::One,
                        alpha_destination: BlendFactor::One,
                    },
                ),
            )
            .render_pass(lighting_pass.clone())
            .build(device.clone())
            .expect("Failed to create pipeline");

        let light_obj_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<obj_loader::ColoredVertex>())
            .vertex_shader(light_obj_vert.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(light_obj_frag.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .render_pass(lighting_pass.clone())
            .build(device.clone())
            .unwrap();

        let (framebuffers, color_buffer, normal_buffer) =
            create_framebuffer(&swapchain_images, render_pass.clone(), &memory_allocator);

        let dummy_verts = CpuAccessibleBuffer::from_iter(
            &memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            obj_loader::DummyVertex::list().iter().cloned(),
        )
        .unwrap();

        let ambient_buffer = CpuAccessibleBuffer::from_data(
            &memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            shaders::ambient_frag::ty::AmbientLightData {
                color: [1.0, 1.0, 1.0],
                intensity: 0.1,
            },
        )
        .unwrap();

        let vp = crate::mvp::VP::from_surface(&surface);
        let vp_buffer = create_uniform_buffer(&vp, memory_allocator.clone());
        let vp_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
        let vp_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, vp_buffer.clone())],
        )
        .unwrap();

        RenderSystem {
            instance,
            surface,
            device,
            queue,
            swapchain,
            swapchain_images,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            model_uniform_buffer_pool,
            directional_uniform_buffer_pool,
            render_pass,
            deferred_pipeline,
            directional_pipeline,
            ambient_pipeline,
            light_obj_pipeline,
            framebuffers,
            color_buffer,
            normal_buffer,
            dummy_verts,
            ambient_buffer,
            vp,
            vp_buffer,
            vp_set,
            render_stage: RenderStage::Stopped,
            commands: None,
            image_index: 0,
            acquire_future: None,
        }
    }

    pub fn set_view(&mut self, view: &glam::Mat4) {
        self.vp.view = *view;
        self.vp_buffer = create_uniform_buffer(&self.vp, self.memory_allocator.clone());

        let vp_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        self.vp_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.vp_buffer.clone())],
        )
        .unwrap();
    }

    pub fn start_frame(&mut self) {
        match self.render_stage {
            RenderStage::Stopped => {
                self.render_stage = RenderStage::Deferred;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let (image_index, suboptimal, acquire_future) =
            match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain();
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            self.recreate_swapchain();
            return;
        }

        let clear_values = vec![
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some([0.0, 0.0, 0.0, 1.0].into()),
            Some(1.0.into()),
        ];

        let mut commands = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        commands
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassContents::Inline,
            )
            .unwrap();

        self.commands = Some(commands);
        self.image_index = image_index;
        self.acquire_future = Some(acquire_future);
    }

    pub fn render_model(&mut self, model: &mut obj_loader::Model) {
        match self.render_stage {
            RenderStage::Deferred => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let model_subbuffer = {
            let (model_mat, normal_mat) = (model.model_matrix(), model.normal_matrix());

            let uniform_data = shaders::deferred_vert::ty::ModelData {
                model: model_mat.to_cols_array_2d(),
                normals: normal_mat.to_cols_array_2d(),
            };

            self.model_uniform_buffer_pool
                .from_data(uniform_data)
                .unwrap()
        };

        let model_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(1)
            .unwrap();
        let model_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            model_layout.clone(),
            [WriteDescriptorSet::buffer(0, model_subbuffer.clone())],
        )
        .unwrap();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            model.data().iter().cloned(),
        )
        .unwrap();

        let view_port = self.view_port_from_surface();
        self.commands
            .as_mut()
            .unwrap()
            .set_viewport(0, [view_port])
            .bind_pipeline_graphics(self.deferred_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.deferred_pipeline.layout().clone(),
                0,
                (self.vp_set.clone(), model_set.clone()),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
    }

    pub fn render_directional(&mut self, directional_light: &light::DirectionalLight) {
        match self.render_stage {
            RenderStage::Ambient => {
                self.render_stage = RenderStage::Directional;
            }
            RenderStage::Directional => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let directional_subbuffer = self.generate_directional_buffer(directional_light);

        let directional_layout = self
            .directional_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        let directional_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            directional_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::buffer(2, directional_subbuffer.clone()),
            ],
        )
        .unwrap();

        let view_port = self.view_port_from_surface();
        self.commands
            .as_mut()
            .unwrap()
            .set_viewport(0, [view_port])
            .bind_pipeline_graphics(self.directional_pipeline.clone())
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.directional_pipeline.layout().clone(),
                0,
                directional_set.clone(),
            )
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();
    }

    pub fn render_ambient(&mut self) {
        match self.render_stage {
            RenderStage::Deferred => {
                self.render_stage = RenderStage::Ambient;
            }
            RenderStage::Ambient => {
                return;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let ambient_layout = self.ambient_pipeline.layout().set_layouts().get(0).unwrap();
        let ambient_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            ambient_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.ambient_buffer.clone()),
            ],
        )
        .unwrap();

        let view_port = self.view_port_from_surface();
        self.commands
            .as_mut()
            .unwrap()
            .next_subpass(SubpassContents::Inline)
            .unwrap()
            .bind_pipeline_graphics(self.ambient_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.ambient_pipeline.layout().clone(),
                0,
                ambient_set.clone(),
            )
            .set_viewport(0, [view_port])
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();
    }

    pub fn render_light_object(&mut self, directional_light: &light::DirectionalLight) {
        match self.render_stage {
            RenderStage::Directional => {
                self.render_stage = RenderStage::LightObject;
            }
            RenderStage::LightObject => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let mut model = obj_loader::Model::builder("models/sphere.obj")
            .color(directional_light.color)
            .uniform_scale_factor(0.2)
            .build();

        model.translate(directional_light.get_position());

        let model_subbuffer = {
            let (model_mat, normal_mat) = (model.model_matrix(), model.normal_matrix());

            let uniform_data = shaders::deferred_vert::ty::ModelData {
                model: model_mat.to_cols_array_2d(),
                normals: normal_mat.to_cols_array_2d(),
            };

            self.model_uniform_buffer_pool
                .from_data(uniform_data)
                .unwrap()
        };

        let model_layout = self
            .light_obj_pipeline
            .layout()
            .set_layouts()
            .get(1)
            .unwrap();
        let model_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            model_layout.clone(),
            [WriteDescriptorSet::buffer(0, model_subbuffer.clone())],
        )
        .unwrap();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            model.color_data().iter().cloned(),
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .bind_pipeline_graphics(self.light_obj_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.light_obj_pipeline.layout().clone(),
                0,
                (self.vp_set.clone(), model_set.clone()),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
    }

    pub fn finish_frame(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
        match self.render_stage {
            RenderStage::Directional => {}
            RenderStage::LightObject => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let mut commands = self.commands.take().unwrap();
        commands.end_render_pass().unwrap();
        let command_buffer = commands.build().unwrap();

        let af = self.acquire_future.take().unwrap();

        let mut local_future: Option<Box<dyn GpuFuture>> =
            Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<dyn GpuFuture>);

        std::mem::swap(&mut local_future, previous_frame_end);

        let future = local_future
            .take()
            .unwrap()
            .join(af)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    self.image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                *previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain();
                *previous_frame_end =
                    Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                *previous_frame_end =
                    Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
        }

        self.commands = None;
        self.render_stage = RenderStage::Stopped;
    }

    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_buffer = CpuAccessibleBuffer::from_data(
            &self.memory_allocator,
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            shaders::ambient_frag::ty::AmbientLightData { color, intensity },
        )
        .unwrap();
    }

    pub fn recreate_swapchain(&mut self) {
        self.vp.projection = crate::mvp::VP::from_surface(&self.surface).projection;

        let (new_swapchain, new_images) = crate::setup::create_swapchain_and_images(
            self.device.clone(),
            self.surface.clone(),
            Some(self.swapchain.clone()),
        );
        let (new_framebuffers, new_color_buffer, new_normal_buffer) = create_framebuffer(
            &new_images,
            self.render_pass.clone(),
            &self.memory_allocator,
        );

        self.swapchain = new_swapchain;
        self.framebuffers = new_framebuffers;
        self.color_buffer = new_color_buffer;
        self.normal_buffer = new_normal_buffer;

        self.vp_buffer = create_uniform_buffer(&self.vp, self.memory_allocator.clone());

        let vp_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        self.vp_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.vp_buffer.clone())],
        )
        .unwrap();
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    fn view_port_from_surface(&self) -> Viewport {
        Viewport {
            origin: [0.0, 0.0],
            dimensions: crate::setup::surface_extent(&self.surface).into(),
            depth_range: 0.0..1.0,
        }
    }

    fn generate_directional_buffer(
        &self,
        light: &light::DirectionalLight,
    ) -> Arc<CpuBufferPoolSubbuffer<shaders::directional_frag::ty::DirectionalLightData>> {
        use glam::swizzles::Vec3Swizzles;
        let position = glam::Vec3::from_array(light.position);
        let uniform_data = shaders::directional_frag::ty::DirectionalLightData {
            position: position.xyzz().into(),
            color: light.color,
        };

        self.directional_uniform_buffer_pool
            .from_data(uniform_data)
            .unwrap()
    }
}

#[allow(clippy::type_complexity)]
fn create_framebuffer(
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

pub fn create_uniform_buffer(
    vp: &crate::mvp::VP,
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
            view: vp.view.to_cols_array_2d(),
            projection: vp.projection.to_cols_array_2d(),
        },
    )
    .expect("Failed to create VP buffer")
}
