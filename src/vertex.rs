use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct VertexA {
    pub position: [f32; 3],
    pub color: [f32; 3],
}
vulkano::impl_vertex!(VertexA, position, color);
