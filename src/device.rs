use std::sync::Arc;

use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo,
    },
    instance::Instance,
    swapchain::Surface,
};

pub struct DeviceAndQueue {
    logical: Arc<Device>,
    queues: Vec<Arc<Queue>>,
}

impl DeviceAndQueue {
    pub fn new_for_window_app(instance: Arc<Instance>, surface: Arc<Surface>) -> Self {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true, // required to create swapchain
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate available physical device")
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
            })
            .expect("Failed to find physical device suitable for window app");

        let (logical_device, queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("Failed to create vulkan logical device");
        let queues = queues.collect::<Vec<Arc<Queue>>>();
        assert!(!queues.is_empty(), "Failed to get suitable queues");

        Self {
            logical: logical_device,
            queues,
        }
    }

    pub fn get_device_and_first_queue(&self) -> (Arc<Device>, Arc<Queue>) {
        (self.logical.clone(), self.queues[0].clone())
    }
}
