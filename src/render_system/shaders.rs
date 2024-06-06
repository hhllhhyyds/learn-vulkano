pub(super) mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/render_system/shaders/deferred.frag"
    }
}

pub(super) mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/render_system/shaders/deferred.vert",
        types_meta: { #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)] },
    }
}

pub(super) mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/render_system/shaders/directional.frag",
        types_meta: { #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)] },
    }
}

pub(super) mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/render_system/shaders/directional.vert",
    }
}

pub(super) mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/render_system/shaders/ambient.vert",
    }
}

pub(super) mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/render_system/shaders/ambient.frag",
        types_meta: { #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)] },
    }
}

pub(super) mod light_obj_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/render_system/shaders/light_obj.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub(super) mod light_obj_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/render_system/shaders/light_obj.frag"
    }
}
