use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct VertexA {
    pub position: [f32; 3],
    pub color: [f32; 3],
}
vulkano::impl_vertex!(VertexA, position, color);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct VertexB {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}
vulkano::impl_vertex!(VertexB, position, normal, color);

pub const CUBE_B: [VertexB; 36] = [
    // front face
    VertexB {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, 1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, 1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, -1.000000, 1.000000],
        normal: [0.0000, 0.0000, 1.0000],
        color: [1.0, 0.35, 0.137],
    },
    // back face
    VertexB {
        position: [1.000000, -1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, 1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, -1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, -1.000000, -1.000000],
        normal: [0.0000, 0.0000, -1.0000],
        color: [1.0, 0.35, 0.137],
    },
    // top face
    VertexB {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, -1.000000, 1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, -1.000000, -1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, -1.000000, -1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, -1.000000, -1.000000],
        normal: [0.0000, -1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    // bottom face
    VertexB {
        position: [1.000000, 1.000000, 1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, 1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, 1.000000, 1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, 1.000000, -1.000000],
        normal: [0.0000, 1.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    // left face
    VertexB {
        position: [-1.000000, -1.000000, -1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, -1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, 1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, -1.000000, -1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, 1.000000, 1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [-1.000000, -1.000000, 1.000000],
        normal: [-1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    // right face
    VertexB {
        position: [1.000000, -1.000000, 1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, 1.000000, 1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, 1.000000, -1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, -1.000000, 1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, 1.000000, -1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
    VertexB {
        position: [1.000000, -1.000000, -1.000000],
        normal: [1.0000, 0.0000, 0.0000],
        color: [1.0, 0.35, 0.137],
    },
];
