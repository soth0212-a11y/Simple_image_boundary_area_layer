use wgpu;

pub struct L1Pipelines {
    pub layout: wgpu::BindGroupLayout,
    pub init_pipeline: wgpu::ComputePipeline,
    pub prop_pipeline: wgpu::ComputePipeline,
}

pub struct L1Buffers {
    pub w: u32,
    pub h: u32,
    pub params0: wgpu::Buffer,
    pub plane_a: wgpu::Buffer,
    pub plane_b: wgpu::Buffer,
    pub edge4_out: wgpu::Buffer,
    pub stop4_out: wgpu::Buffer,
}

pub fn default_iters(w: u32, h: u32, config_iter: u32) -> u32 {
    let max_dim = w.max(h);
    let base = if config_iter == 0 { max_dim } else { config_iter };
    base.min(max_dim).min(128).max(1)
}

pub fn build_l1_pipelines(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> L1Pipelines {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l1_plane_label_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l1_plane_label_pipeline_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l1_init_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l1_init"),
        compilation_options: Default::default(),
        cache: None,
    });

    let prop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l1_propagate_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l1_propagate"),
        compilation_options: Default::default(),
        cache: None,
    });

    L1Pipelines {
        layout,
        init_pipeline,
        prop_pipeline,
    }
}

pub fn ensure_l1_buffers(device: &wgpu::Device, w: u32, h: u32) -> L1Buffers {
    let params0 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_params0"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let cell_count = (w * h).max(1) as u64;
    let plane_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_plane_a"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let plane_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_plane_b"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let edge4_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_edge4_out"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stop4_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_stop4_out"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    L1Buffers {
        w,
        h,
        params0,
        plane_a,
        plane_b,
        edge4_out,
        stop4_out,
    }
}

pub fn dispatch_l1(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &L1Pipelines,
    bufs: &L1Buffers,
    cell_rgb: &wgpu::Buffer,
    edge4_in: &wgpu::Buffer,
    th: u32,
    iters: u32,
) -> bool {
    let params0 = [bufs.w, bufs.h, th, iters];
    queue.write_buffer(&bufs.params0, 0, bytemuck::cast_slice(&params0));

    let bg_init = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l1_bg_init"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cell_rgb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: edge4_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.plane_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.plane_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.edge4_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.stop4_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.params0.as_entire_binding() },
        ],
    });

    let bg_prop_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l1_bg_prop_ab"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cell_rgb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: edge4_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.plane_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.plane_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.edge4_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.stop4_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.params0.as_entire_binding() },
        ],
    });
    let bg_prop_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l1_bg_prop_ba"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cell_rgb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: edge4_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.plane_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.plane_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.edge4_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.stop4_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.params0.as_entire_binding() },
        ],
    });

    let dispatch_x = (bufs.w + 15) / 16;
    let dispatch_y = (bufs.h + 15) / 16;

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l1_plane_label_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l1_init_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.init_pipeline);
        pass.set_bind_group(0, &bg_init, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    let mut in_a = true;
    for _ in 0..iters {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l1_prop_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.prop_pipeline);
        if in_a {
            pass.set_bind_group(0, &bg_prop_ab, &[]);
        } else {
            pass.set_bind_group(0, &bg_prop_ba, &[]);
        }
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        in_a = !in_a;
    }

    queue.submit([enc.finish()]);
    in_a
}
