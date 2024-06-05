use std::{f64::consts::PI, time::Instant};

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

    let mut suzanne = Model::new("models/suzanne.obj").build();
    suzanne.translate(glam::vec3(5.0, 2.0, -6.0));

    let mut torus = Model::new("models/torus.obj").build();
    torus.translate(glam::vec3(0.0, -2.0, -5.0));

    let directional_light_r = DirectionalLight {
        position: [-4.0, -4.0, 0.0],
        color: [1.0, 0.0, 0.0],
    };
    let directional_light_g = DirectionalLight {
        position: [4.0, -4.0, 0.0],
        color: [0.0, 1.0, 0.0],
    };
    let directional_light_b = DirectionalLight {
        position: [0.0, 4.0, 0.0],
        color: [0.0, 0.0, 1.0],
    };

    let mut previous_frame_end =
        Some(Box::new(vulkano::sync::now(system.device())) as Box<dyn GpuFuture>);

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
            system.recreate_swapchain();
        }
        Event::RedrawEventsCleared => {
            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            let elapsed = rotation_start.elapsed().as_secs() as f64
                + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
            let elapsed_as_radians = elapsed * PI / 180.0;

            teapot.zero_rotation();
            teapot.rotate(elapsed_as_radians as f32 * 50.0, glam::vec3(0.0, 0.0, 1.0));
            teapot.rotate(elapsed_as_radians as f32 * 30.0, glam::vec3(0.0, 1.0, 0.0));
            teapot.rotate(elapsed_as_radians as f32 * 20.0, glam::vec3(1.0, 0.0, 0.0));

            suzanne.zero_rotation();
            suzanne.rotate(elapsed_as_radians as f32 * 25.0, glam::vec3(0.0, 0.0, 1.0));
            suzanne.rotate(elapsed_as_radians as f32 * 10.0, glam::vec3(0.0, 1.0, 0.0));
            suzanne.rotate(elapsed_as_radians as f32 * 60.0, glam::vec3(1.0, 0.0, 0.0));

            torus.zero_rotation();
            torus.rotate(elapsed_as_radians as f32 * 5.0, glam::vec3(0.0, 0.0, 1.0));
            torus.rotate(elapsed_as_radians as f32 * 45.0, glam::vec3(0.0, 1.0, 0.0));
            torus.rotate(elapsed_as_radians as f32 * 12.0, glam::vec3(1.0, 0.0, 0.0));

            system.start_frame();
            system.render_model(&mut teapot);
            system.render_model(&mut suzanne);
            system.render_model(&mut torus);
            system.render_ambient();
            system.render_directional(&directional_light_r);
            system.render_directional(&directional_light_g);
            system.render_directional(&directional_light_b);
            system.finish_frame(&mut previous_frame_end);
        }
        _ => {}
    });
}
