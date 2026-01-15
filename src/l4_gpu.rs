use wgpu;

pub struct L4ChannelsPipelines {
    pub layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
}

pub struct L4ChannelsBuffers {
    pub w: u32,
    pub h: u32,
    pub dims: wgpu::Buffer,
    pub bbox0: wgpu::Buffer,
    pub bbox1: wgpu::Buffer,
    pub meta: wgpu::Buffer,
}

pub fn build_l4_channels_pipelines(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> L4ChannelsPipelines {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l4_channels_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l4_channels_pipeline_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l4_channels_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l4_build_bboxes"),
        compilation_options: Default::default(),
        cache: None,
    });

    L4ChannelsPipelines { layout, pipeline }
}

pub fn ensure_l4_channels_buffers(device: &wgpu::Device, w: u32, h: u32) -> L4ChannelsBuffers {
    let dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_dims"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_size = (w * h).max(1) as u64 * 4;
    let bbox0 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_bbox0"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_bbox1"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let meta = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_meta"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    L4ChannelsBuffers { w, h, dims, bbox0, bbox1, meta }
}

pub fn dispatch_l4_channels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &L4ChannelsPipelines,
    bufs: &L4ChannelsBuffers,
    s_active: &wgpu::Buffer,
    conn8: &wgpu::Buffer,
) {
    let dims_data = [bufs.w, bufs.h, 0u32, 0u32];
    queue.write_buffer(&bufs.dims, 0, bytemuck::cast_slice(&dims_data));

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l4_channels_bg"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: conn8.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.bbox0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.bbox1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.meta.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.dims.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l4_channels_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l4_channels_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((bufs.w + 15) / 16, (bufs.h + 15) / 16, 1);
    }
    queue.submit([enc.finish()]);
}
