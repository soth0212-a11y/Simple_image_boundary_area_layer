use wgpu;

pub struct L3Pipelines {
    pub pool_layout: wgpu::BindGroupLayout,
    pub pool: wgpu::ComputePipeline,
    pub conn_layout: wgpu::BindGroupLayout,
    pub conn: wgpu::ComputePipeline,
}

pub struct L3Buffers {
    pub in_w: u32,
    pub in_h: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub dims: wgpu::Buffer,
    pub s_active: wgpu::Buffer,
    pub conn8: wgpu::Buffer,
}

pub fn build_l3_pipelines(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> L3Pipelines {
    let pool_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l3_pool_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pool_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l3_pool_pipeline_layout"),
        bind_group_layouts: &[&pool_layout],
        push_constant_ranges: &[],
    });
    let pool = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l3_pool_pipeline"),
        layout: Some(&pool_pipeline_layout),
        module: &shader_module,
        entry_point: Some("l3_stride2_pool"),
        compilation_options: Default::default(),
        cache: None,
    });

    let conn_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l3_conn_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let conn_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l3_conn_pipeline_layout"),
        bind_group_layouts: &[&conn_layout],
        push_constant_ranges: &[],
    });
    let conn = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l3_conn_pipeline"),
        layout: Some(&conn_pipeline_layout),
        module: &shader_module,
        entry_point: Some("l3_conn8"),
        compilation_options: Default::default(),
        cache: None,
    });

    L3Pipelines {
        pool_layout,
        pool,
        conn_layout,
        conn,
    }
}

pub fn ensure_l3_buffers(device: &wgpu::Device, in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> L3Buffers {
    let dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_dims"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_size = (out_w * out_h).max(1) as u64 * 4;
    let s_active = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_s_active"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let conn8 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_conn8"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    L3Buffers { in_w, in_h, out_w, out_h, dims, s_active, conn8 }
}

pub fn dispatch_l3(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &L3Pipelines,
    bufs: &L3Buffers,
    in_active: &wgpu::Buffer,
) {
    let dims_data = [bufs.in_w, bufs.in_h, bufs.out_w, bufs.out_h];
    queue.write_buffer(&bufs.dims, 0, bytemuck::cast_slice(&dims_data));

    let pool_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l3_pool_bg"),
        layout: &pipelines.pool_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bufs.dims.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: in_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.s_active.as_entire_binding() },
        ],
    });

    let conn_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l3_conn_bg"),
        layout: &pipelines.conn_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bufs.dims.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs.s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.conn8.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l3_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l3_pool_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.pool);
        pass.set_bind_group(0, &pool_bg, &[]);
        pass.dispatch_workgroups((bufs.out_w + 15) / 16, (bufs.out_h + 15) / 16, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l3_conn_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.conn);
        pass.set_bind_group(0, &conn_bg, &[]);
        pass.dispatch_workgroups((bufs.out_w + 15) / 16, (bufs.out_h + 15) / 16, 1);
    }
    queue.submit([enc.finish()]);
}
