#![allow(clippy::all)]

use std::cell::Cell;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

pub struct RawVertex {
    pub vals: [f32; 3],
}

impl RawVertex {
    pub fn new(inpt: &str) -> RawVertex {
        let items = inpt.split_whitespace();
        let mut content: Vec<f32> = Vec::new();
        for item in items {
            content.push(item.parse().unwrap());
        }
        if content.len() == 2 {
            content.push(0.0);
        }
        RawVertex {
            vals: [
                *content.get(0).unwrap(),
                *content.get(1).unwrap(),
                *content.get(2).unwrap(),
            ],
        }
    }
}

pub struct RawFace {
    pub verts: [usize; 3],
    pub norms: Option<[usize; 3]>,
    pub text: Option<[usize; 3]>,
}

impl RawFace {
    // call with invert = true if the models are using a clockwise winding order
    //
    // Blender files are a common example
    pub fn new(raw_arg: &str, invert: bool) -> RawFace {
        let arguments: Vec<&str> = raw_arg.split_whitespace().collect();
        RawFace {
            verts: RawFace::parse(arguments.clone(), 0, invert).unwrap(),
            norms: RawFace::parse(arguments.clone(), 2, invert),
            text: RawFace::parse(arguments.clone(), 1, invert),
        }
    }

    fn parse(inpt: Vec<&str>, index: usize, invert: bool) -> Option<[usize; 3]> {
        let f1: Vec<&str> = inpt.get(0).unwrap().split("/").collect();
        let f2: Vec<&str> = inpt.get(1).unwrap().split("/").collect();
        let f3: Vec<&str> = inpt.get(2).unwrap().split("/").collect();
        let a1 = f1.get(index).unwrap().to_string();
        let a2 = f2.get(index).unwrap().to_string();
        let a3 = f3.get(index).unwrap().to_string();
        match a1.as_str() {
            "" => None,
            _ => {
                let p1: usize = a1.parse().unwrap();
                let (p2, p3): (usize, usize) = if invert {
                    (a3.parse().unwrap(), a2.parse().unwrap())
                } else {
                    (a2.parse().unwrap(), a3.parse().unwrap())
                };
                Some([p1 - 1, p2 - 1, p3 - 1]) // .obj files aren't 0-index
            }
        }
    }
}

impl std::fmt::Display for RawFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let verts = format!("{}/{}/{}", self.verts[0], self.verts[1], self.verts[2]);
        let normals = match self.norms {
            None => "None".to_string(),
            Some(x) => {
                format!("{}/{}/{}", x[0], x[1], x[2])
            }
        };
        let textures = match self.text {
            None => "None".to_string(),
            Some(x) => {
                format!("{}/{}/{}", x[0], x[1], x[2])
            }
        };
        write!(
            f,
            "Face:\n\tVertices: {}\n\tNormals: {}\n\tTextures: {}",
            verts, normals, textures
        )
    }
}

pub struct Loader {
    color: [f32; 3],
    verts: Vec<RawVertex>,
    norms: Vec<RawVertex>,
    #[allow(unused)]
    text: Vec<RawVertex>,
    faces: Vec<RawFace>,
    #[allow(unused)]
    invert_winding_order: bool,
}

impl Loader {
    pub fn new(file_name: &str, custom_color: [f32; 3], invert_winding_order: bool) -> Loader {
        let color = custom_color;
        let input = File::open(file_name).unwrap();
        let buffered = BufReader::new(input);
        let mut verts: Vec<RawVertex> = Vec::new();
        let mut norms: Vec<RawVertex> = Vec::new();
        let mut text: Vec<RawVertex> = Vec::new();
        let mut faces: Vec<RawFace> = Vec::new();
        for raw_line in buffered.lines() {
            let line = raw_line.unwrap();
            if line.len() > 2 {
                match line.split_at(2) {
                    ("v ", x) => {
                        verts.push(RawVertex::new(x));
                    }
                    ("vn", x) => {
                        norms.push(RawVertex::new(x));
                    }
                    ("vt", x) => {
                        text.push(RawVertex::new(x));
                    }
                    ("f ", x) => {
                        faces.push(RawFace::new(x, invert_winding_order));
                    }
                    (_, _) => {}
                };
            }
        }
        Loader {
            color,
            verts,
            norms,
            text,
            faces,
            invert_winding_order,
        }
    }

    pub fn as_normal_vertices(&self) -> Vec<NormalVertex> {
        let mut ret: Vec<NormalVertex> = Vec::new();
        for face in &self.faces {
            let verts = face.verts;
            let normals = face.norms.unwrap();
            ret.push(NormalVertex {
                position: self.verts.get(verts[0]).unwrap().vals,
                normal: self.norms.get(normals[0]).unwrap().vals,
                color: self.color,
            });
            ret.push(NormalVertex {
                position: self.verts.get(verts[1]).unwrap().vals,
                normal: self.norms.get(normals[1]).unwrap().vals,
                color: self.color,
            });
            ret.push(NormalVertex {
                position: self.verts.get(verts[2]).unwrap().vals,
                normal: self.norms.get(normals[2]).unwrap().vals,
                color: self.color,
            });
        }
        ret
    }
}

/// A vertex type intended to be used to provide dummy rendering
/// data for rendering passes that do not require geometry data.
/// This is due to a quirk of the Vulkan API in that *all*
/// render passes require some sort of input.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct DummyVertex {
    /// A regular position vector with the z-value shaved off for space.
    /// This assumes the shaders will take a `vec2` and transform it as
    /// needed.
    pub position: [f32; 2],
}
vulkano::impl_vertex!(DummyVertex, position);

impl DummyVertex {
    /// Creates an array of `DummyVertex` values.
    ///
    /// This is intended to compliment the use of this data type for passing to
    /// deferred rendering passes that do not actually require geometry input.
    /// This list will draw a square across the entire rendering area. This will
    /// cause the fragment shaders to execute on all pixels in the rendering
    /// area.
    ///
    /// # Example
    ///
    /// ```glsl
    /// #version 450
    ///
    ///layout(location = 0) in vec2 position;
    ///
    ///void main() {
    ///    gl_Position = vec4(position, 0.0, 1.0);
    ///}
    /// ```
    pub fn list() -> [DummyVertex; 6] {
        [
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [-1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, -1.0],
            },
        ]
    }
}

/// A structure for the vertex information used in earlier lessons
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct ColoredVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}
vulkano::impl_vertex!(ColoredVertex, position, color);

/// A structure used for the vertex information starting
/// from our lesson on lighting
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct NormalVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}
vulkano::impl_vertex!(NormalVertex, position, normal, color);

impl fmt::Display for DummyVertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = format!("[{:.6}, {:.6}]", self.position[0], self.position[1]);
        write!(f, "DummyVertex {{ position: {} }}", pos)
    }
}

impl fmt::Display for ColoredVertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = format!(
            "[{:.6}, {:.6}, {:.6}]",
            self.position[0], self.position[1], self.position[2]
        );
        let color = format!(
            "[{:.6}, {:.6}, {:.6}]",
            self.color[0], self.color[1], self.color[2]
        );
        write!(f, "ColoredVertex {{ position: {}, color: {} }}", pos, color)
    }
}

impl fmt::Display for NormalVertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = format!(
            "[{:.6}, {:.6}, {:.6}]",
            self.position[0], self.position[1], self.position[2]
        );
        let color = format!(
            "[{:.6}, {:.6}, {:.6}]",
            self.color[0], self.color[1], self.color[2]
        );
        let norms = format!(
            "[{:.6}, {:.6}, {:.6}]",
            self.normal[0], self.normal[1], self.normal[2]
        );
        write!(
            f,
            "NormalVertex {{ position: {}, normal: {}, color: {} }}",
            pos, norms, color
        )
    }
}

/// Holds our data for a renderable model, including the model matrix data
///
/// Note: When building an instance of `Model` the loader will assume that
/// the input obj file is in clockwise winding order. If it is already in
/// counter-clockwise winding order, call `.invert_winding_order(false)`
/// when building the `Model`.
pub struct Model {
    data: Vec<NormalVertex>,
    translation: Mat4,
    rotation: Mat4,
    uniform_scale: f32,

    // We might call multiple translation/rotation calls
    // in between asking for the model matrix. This lets us
    // only recreate the model matrices when needed.
    // Use a Cell with the interior mutability pattern,
    // so that it can be modified by methods that don't take &mut self
    cache: Cell<Option<(ModelMatrices, NormalMatrices)>>,
}

#[derive(Copy, Clone)]
struct ModelMatrices {
    model: Mat4,
}

#[derive(Copy, Clone)]
struct NormalMatrices {
    normal: Mat4,
}

pub struct ModelBuilder {
    file_name: String,
    custom_color: [f32; 3],
    invert: bool,
    scale_factor: f32,
}

impl ModelBuilder {
    fn new(file: String) -> ModelBuilder {
        ModelBuilder {
            file_name: file,
            custom_color: [1.0, 0.35, 0.137],
            invert: true,
            scale_factor: 1.0,
        }
    }

    pub fn build(self) -> Model {
        let loader = Loader::new(self.file_name.as_str(), self.custom_color, self.invert);
        Model {
            data: loader.as_normal_vertices(),
            translation: Mat4::IDENTITY,
            rotation: Mat4::IDENTITY,
            uniform_scale: self.scale_factor,
            cache: Cell::new(None),
        }
    }

    pub fn color(mut self, new_color: [f32; 3]) -> ModelBuilder {
        self.custom_color = new_color;
        self
    }

    pub fn file(mut self, file: String) -> ModelBuilder {
        self.file_name = file;
        self
    }

    pub fn invert_winding_order(mut self, invert: bool) -> ModelBuilder {
        self.invert = invert;
        self
    }

    pub fn uniform_scale_factor(mut self, scale: f32) -> ModelBuilder {
        self.scale_factor = scale;
        self
    }
}

impl Model {
    pub fn new(file_name: &str) -> ModelBuilder {
        ModelBuilder::new(file_name.into())
    }

    pub fn data(&self) -> Vec<NormalVertex> {
        self.data.clone()
    }

    pub fn color_data(&self) -> Vec<ColoredVertex> {
        let mut ret: Vec<ColoredVertex> = Vec::new();
        for v in &self.data {
            ret.push(ColoredVertex {
                position: v.position,
                color: v.color,
            });
        }
        ret
    }

    pub fn model_matrix(&self) -> Mat4 {
        if let Some(cache) = self.cache.get() {
            return cache.0.model;
        }

        // recalculate matrix
        let model = self.translation * self.rotation;
        let model = model * Mat4::from_scale([self.uniform_scale; 3].into());
        let normal = model.inverse().transpose();

        self.cache
            .set(Some((ModelMatrices { model }, NormalMatrices { normal })));

        model
    }

    pub fn normal_matrix(&self) -> Mat4 {
        if let Some(cache) = self.cache.get() {
            return cache.1.normal;
        }

        // recalculate matrix
        let model = self.translation * self.rotation;
        let model = model * Mat4::from_scale([self.uniform_scale; 3].into());
        let normal = model.inverse().transpose();

        self.cache
            .set(Some((ModelMatrices { model }, NormalMatrices { normal })));

        normal
    }

    pub fn rotate(&mut self, radians: f32, v: Vec3) {
        self.rotation *= Mat4::from_axis_angle(v, radians);
        self.cache.set(None);
    }

    pub fn translate(&mut self, v: Vec3) {
        self.translation *= Mat4::from_translation(v);
        self.cache.set(None);
    }

    /// Return the model's rotation to 0
    pub fn zero_rotation(&mut self) {
        self.rotation = Mat4::IDENTITY;
        self.cache.set(None);
    }
}
