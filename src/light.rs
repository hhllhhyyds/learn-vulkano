use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Pod, Zeroable)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Pod, Zeroable)]
pub struct DirectionalLight {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl DirectionalLight {
    pub fn get_position(&self) -> glam::Vec3 {
        glam::vec3(self.position[0], self.position[1], self.position[2])
    }
}
