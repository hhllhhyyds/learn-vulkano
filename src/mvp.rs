use bytemuck::{Pod, Zeroable};
use glam::Mat4;

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
}

impl Default for VP {
    fn default() -> Self {
        Self::new()
    }
}
