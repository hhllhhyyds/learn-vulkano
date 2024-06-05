use std::{f32::consts::FRAC_PI_2, sync::Arc};

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    memory::allocator::StandardMemoryAllocator,
    swapchain::Surface,
};

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
        let image_extent: [u32; 2] = super::swapchain::surface_extent(surface).into();
        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        let projection = Mat4::perspective_rh_gl(FRAC_PI_2, aspect_ratio, 0.01, 100.0);
        Self {
            view: Mat4::IDENTITY,
            projection,
        }
    }

    pub fn create_uniform_buffer(
        &self,
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
                view: self.view.to_cols_array_2d(),
                projection: self.projection.to_cols_array_2d(),
            },
        )
        .expect("Failed to create VP buffer")
    }
}

impl Default for VP {
    fn default() -> Self {
        Self::new()
    }
}
