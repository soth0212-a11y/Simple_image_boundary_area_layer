use wgpu::{
    self,
    // wgc::device::{self, queue},
};

pub async fn gpu_init() -> (wgpu::Device, wgpu::Adapter, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("wgpu - Failed to create adapter");

    // println!("Running on Adapter: {:#?}", adapter.get_info());

    let mut limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 8 {
        panic!(
            "wgpu - max_storage_buffers_per_shader_stage {} is too low for L1 outputs",
            limits.max_storage_buffers_per_shader_stage
        );
    }
    limits.max_storage_buffers_per_shader_stage = 8;

    let required_features = wgpu::Features::empty();

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features,
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");
    return (device, adapter, queue);
}
