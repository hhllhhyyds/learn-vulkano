use std::f32::consts::FRAC_PI_2;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use vulkano::swapchain::Surface;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MVP {
    pub model: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
}

impl MVP {
    pub fn new() -> MVP {
        MVP {
            model: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
        }
    }
}

impl Default for MVP {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VP {
    pub view: Mat4,
    pub projection: Mat4,
}

impl VP {
    pub fn new() -> VP {
        VP {
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
        }
    }

    pub fn from_surface(surface: &Surface) -> Self {
        let image_extent: [u32; 2] = crate::setup::surface_extent(surface).into();
        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        let projection = Mat4::perspective_rh_gl(FRAC_PI_2, aspect_ratio, 0.01, 100.0);
        Self {
            view: Mat4::IDENTITY,
            projection,
        }
    }
}

impl Default for VP {
    fn default() -> Self {
        Self::new()
    }
}
