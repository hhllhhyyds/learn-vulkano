use std::sync::Arc;

use vulkano::instance::{Instance, InstanceCreateInfo, InstanceCreationError};
use vulkano::{Version, VulkanLibrary};

pub fn instance_for_window_requirements() -> Result<Arc<Instance>, InstanceCreationError> {
    let library = VulkanLibrary::new().expect("Failed to load vulkan library");
    let extensions = vulkano_win::required_extensions(&library);

    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: extensions,
            enumerate_portability: true, // required for MoltenVK on macOS
            max_api_version: Some(Version::V1_1),
            ..Default::default()
        },
    )
}
