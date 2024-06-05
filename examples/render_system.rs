use learn_vulkano::render_system::{light::DirectionalLight, obj_loader::Model, RenderSystem};

use vulkano::sync::GpuFuture;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

fn main() {
    let event_loop = EventLoop::new();
    let mut system = RenderSystem::new(&event_loop);

    // system.set_view(&glam::Mat4::from_mat3(glam::Mat3 {
    //     x_axis: glam::vec3(0.0, 0.0, 0.1),
    //     y_axis: glam::vec3(0.0, 0.0, 0.0),
    //     z_axis: glam::vec3(0.0, 1.0, 0.0),
    // }));

    let mut teapot = Model::new("models/teapot.obj").build();
    teapot.translate(glam::vec3(-5.0, 2.0, -8.0));

    let directional_light = DirectionalLight {
        position: [-4.0, -4.0, 0.0],
        color: [1.0, 0.0, 0.0],
    };

    let mut previous_frame_end =
        Some(Box::new(vulkano::sync::now(system.device())) as Box<dyn GpuFuture>);

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
            system.recreate_swapchain();
        }
        Event::RedrawEventsCleared => {
            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            system.start_frame();
            system.render_model(&mut teapot);
            system.render_ambient();
            system.render_directional(&directional_light);
            system.finish_frame(&mut previous_frame_end);
        }
        _ => {}
    });
}
