use wgpu;

pub struct L1BboxPipelines {
    pub layout: wgpu::BindGroupLayout,
    pub stride1: wgpu::ComputePipeline,
    pub stride2: wgpu::ComputePipeline,
}

pub struct L1BboxBuffers {
    pub w: u32,
    pub h: u32,
    pub stride2_w: u32,
    pub stride2_h: u32,
    pub params: wgpu::Buffer,
    pub bbox0_s1: wgpu::Buffer,
    pub bbox1_s1: wgpu::Buffer,
    pub color_s1: wgpu::Buffer,
    pub bbox0_s2: wgpu::Buffer,
    pub bbox1_s2: wgpu::Buffer,
    pub color_s2: wgpu::Buffer,
}

pub fn build_l1_bbox_pipelines(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> L1BboxPipelines {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l1_bbox_layout"),
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
        label: Some("l1_bbox_pipeline_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let stride1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l1_bbox_stride1"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l1_bbox_stride1"),
        compilation_options: Default::default(),
        cache: None,
    });

    let stride2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l1_bbox_stride2"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l1_bbox_stride2"),
        compilation_options: Default::default(),
        cache: None,
    });

    L1BboxPipelines { layout, stride1, stride2 }
}

pub fn ensure_l1_bbox_buffers(device: &wgpu::Device, w: u32, h: u32) -> L1BboxBuffers {
    let stride2_w = (w + 1) / 2;
    let stride2_h = (h + 1) / 2;
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_bbox_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let count_s1 = (w * h).max(1) as u64;
    let bbox0_s1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_bbox0_s1"),
        size: count_s1 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox1_s1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_bbox1_s1"),
        size: count_s1 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let color_s1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_bbox_color_s1"),
        size: count_s1 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let count_s2 = (stride2_w * stride2_h).max(1) as u64;
    let bbox0_s2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_bbox0_s2"),
        size: count_s2 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox1_s2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_bbox1_s2"),
        size: count_s2 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let color_s2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_bbox_color_s2"),
        size: count_s2 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    L1BboxBuffers {
        w,
        h,
        stride2_w,
        stride2_h,
        params,
        bbox0_s1,
        bbox1_s1,
        color_s1,
        bbox0_s2,
        bbox1_s2,
        color_s2,
    }
}

pub fn dispatch_l1_bbox(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &L1BboxPipelines,
    bufs: &L1BboxBuffers,
    s_active: &wgpu::Buffer,
    packed: &wgpu::Buffer,
    enable_stride2: bool,
) {
    let params_s1 = [bufs.w, bufs.h, 1u32, 0u32];
    queue.write_buffer(&bufs.params, 0, bytemuck::cast_slice(&params_s1));

    let bg_s1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l1_bbox_bg_s1"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: packed.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.bbox0_s1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.bbox1_s1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.color_s1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.params.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l1_bbox_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l1_bbox_s1_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.stride1);
        pass.set_bind_group(0, &bg_s1, &[]);
        pass.dispatch_workgroups((bufs.w + 15) / 16, (bufs.h + 15) / 16, 1);
    }

    if enable_stride2 {
        let params_s2 = [bufs.w, bufs.h, 2u32, 0u32];
        queue.write_buffer(&bufs.params, 0, bytemuck::cast_slice(&params_s2));
        let bg_s2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l1_bbox_bg_s2"),
            layout: &pipelines.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: s_active.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: packed.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.bbox0_s2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bufs.bbox1_s2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bufs.color_s2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bufs.params.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l1_bbox_s2_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.stride2);
        pass.set_bind_group(0, &bg_s2, &[]);
        pass.dispatch_workgroups((bufs.stride2_w + 15) / 16, (bufs.stride2_h + 15) / 16, 1);
    }

    queue.submit([enc.finish()]);
}
