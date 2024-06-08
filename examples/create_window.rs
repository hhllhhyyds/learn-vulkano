use std::sync::Arc;

use vulkano::{
    command_buffer::{CommandBufferUsage, RenderPassBeginInfo, SubpassContents},
    image::{view::ImageView, SwapchainImage},
    render_pass::{Framebuffer, FramebufferCreateInfo, FramebufferCreationError, RenderPass},
};

use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use learn_vulkano::setup::RenderBase;

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
    let mut framebuffers = vec![];
    for image in images {
        let view = ImageView::new_default(image.clone()).unwrap();
        framebuffers.push(Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![view],
                ..Default::default()
            },
        )?);
    }
    Ok(framebuffers)
}
