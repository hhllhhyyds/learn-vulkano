use std::sync::Arc;

use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceCreationError, DeviceExtensions, Queue, QueueCreateInfo,
    },
    instance::Instance,
    swapchain::Surface,
    VulkanError,
};

pub struct DeviceAndQueue {
    pub physical: Arc<PhysicalDevice>,
    pub queue_family_index: u32,
    pub logical: Arc<Device>,
    pub queues: Vec<Arc<Queue>>,
}

#[derive(Debug, Clone, Copy)]
pub enum CreationError {
    EnumPhysicalDeviceError(VulkanError),
    FailedFindSuitableError,
    LogicalDeviceCreationError(DeviceCreationError),
}

impl From<VulkanError> for CreationError {
    fn from(value: VulkanError) -> Self {
        CreationError::EnumPhysicalDeviceError(value)
    }
}

impl From<DeviceCreationError> for CreationError {
    fn from(value: DeviceCreationError) -> Self {
        CreationError::LogicalDeviceCreationError(value)
    }
}

pub fn device_and_queue_for_window_requirements(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
) -> Result<DeviceAndQueue, CreationError> {
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = if let Some((physical_device, queue_family_index)) =
        instance
            .enumerate_physical_devices()?
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // pick first queue_familiy_index that handles graphics and can draw on the surface created by winit
                        q.queue_flags.graphics
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| {
                // lower score for preferred device types
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            }) {
        (physical_device, queue_family_index)
    } else {
        return Err(CreationError::FailedFindSuitableError);
    };

    let (device, queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )?;
    let queues = queues.collect::<Vec<Arc<Queue>>>();

    Ok(DeviceAndQueue {
        physical: physical_device,
        queue_family_index,
        logical: device,
        queues,
    })
}
