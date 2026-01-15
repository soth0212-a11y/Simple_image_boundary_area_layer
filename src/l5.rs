use wgpu;

pub struct L5Pipelines {
    pub init_layout: wgpu::BindGroupLayout,
    pub init_pipeline: wgpu::ComputePipeline,
    pub prop_layout: wgpu::BindGroupLayout,
    pub prop_pipeline: wgpu::ComputePipeline,
    pub reset_layout: wgpu::BindGroupLayout,
    pub reset_pipeline: wgpu::ComputePipeline,
    pub accum_layout: wgpu::BindGroupLayout,
    pub accum_pipeline: wgpu::ComputePipeline,
}

pub struct L5Buffers {
    pub w: u32,
    pub h: u32,
    pub dims: wgpu::Buffer,
    pub label_a: wgpu::Buffer,
    pub label_b: wgpu::Buffer,
    pub acc_minx: wgpu::Buffer,
    pub acc_miny: wgpu::Buffer,
    pub acc_maxx: wgpu::Buffer,
    pub acc_maxy: wgpu::Buffer,
    pub acc_count: wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct MergedBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub count: u32,
}

pub fn default_gap() -> u32 {
    0
}

pub fn default_iterations(w: u32, h: u32) -> u32 {
    let m = w.max(h);
    if m == 0 {
        0
    } else {
        m.min(128)
    }
}

pub fn default_min_count() -> u32 {
    2
}

pub fn build_l5_pipelines(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> L5Pipelines {
    let init_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l5_init_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let init_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l5_init_pipeline_layout"),
        bind_group_layouts: &[&init_layout],
        push_constant_ranges: &[],
    });
    let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_init_pipeline"),
        layout: Some(&init_pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_init_labels"),
        compilation_options: Default::default(),
        cache: None,
    });

    let prop_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l5_prop_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let prop_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l5_prop_pipeline_layout"),
        bind_group_layouts: &[&prop_layout],
        push_constant_ranges: &[],
    });
    let prop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_prop_pipeline"),
        layout: Some(&prop_pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_propagate_labels"),
        compilation_options: Default::default(),
        cache: None,
    });

    let reset_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l5_reset_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let reset_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l5_reset_pipeline_layout"),
        bind_group_layouts: &[&reset_layout],
        push_constant_ranges: &[],
    });
    let reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_reset_pipeline"),
        layout: Some(&reset_pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_reset_accumulators"),
        compilation_options: Default::default(),
        cache: None,
    });

    let accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l5_accum_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let accum_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l5_accum_pipeline_layout"),
        bind_group_layouts: &[&accum_layout],
        push_constant_ranges: &[],
    });
    let accum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_accum_pipeline"),
        layout: Some(&accum_pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_accumulate_bboxes"),
        compilation_options: Default::default(),
        cache: None,
    });

    L5Pipelines {
        init_layout,
        init_pipeline,
        prop_layout,
        prop_pipeline,
        reset_layout,
        reset_pipeline,
        accum_layout,
        accum_pipeline,
    }
}

pub fn ensure_l5_buffers(device: &wgpu::Device, w: u32, h: u32) -> L5Buffers {
    let dims = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_dims"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let count = (w * h).max(1) as u64 * 4;
    let label_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_label_a"),
        size: count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let label_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_label_b"),
        size: count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let acc_minx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_minx"),
        size: count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let acc_miny = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_miny"),
        size: count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let acc_maxx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_maxx"),
        size: count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let acc_maxy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_maxy"),
        size: count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let acc_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_count"),
        size: count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    L5Buffers {
        w,
        h,
        dims,
        label_a,
        label_b,
        acc_minx,
        acc_miny,
        acc_maxx,
        acc_maxy,
        acc_count,
    }
}

pub fn dispatch_l5(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &L5Pipelines,
    bufs: &L5Buffers,
    bbox0: &wgpu::Buffer,
    bbox1: &wgpu::Buffer,
    meta: &wgpu::Buffer,
    gap: u32,
    iterations: u32,
) -> bool {
    let dims_data = [bufs.w, bufs.h, gap, iterations];
    queue.write_buffer(&bufs.dims, 0, bytemuck::cast_slice(&dims_data));

    let bg_init = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_init_bg"),
        layout: &pipelines.init_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bbox0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bbox1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: meta.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.dims.as_entire_binding() },
        ],
    });

    let bg_prop_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_prop_ab_bg"),
        layout: &pipelines.prop_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bbox0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bbox1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: meta.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.dims.as_entire_binding() },
        ],
    });

    let bg_prop_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_prop_ba_bg"),
        layout: &pipelines.prop_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bbox0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bbox1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: meta.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.dims.as_entire_binding() },
        ],
    });

    let bg_reset = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_reset_bg"),
        layout: &pipelines.reset_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 5, resource: bufs.dims.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.acc_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.acc_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.acc_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs.acc_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: bufs.acc_count.as_entire_binding() },
        ],
    });

    let bg_accum_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_accum_a_bg"),
        layout: &pipelines.accum_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bbox0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bbox1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.dims.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.acc_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.acc_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.acc_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs.acc_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: bufs.acc_count.as_entire_binding() },
        ],
    });

    let bg_accum_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_accum_b_bg"),
        layout: &pipelines.accum_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bbox0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bbox1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.dims.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.acc_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.acc_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.acc_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs.acc_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: bufs.acc_count.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l5_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_init_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.init_pipeline);
        pass.set_bind_group(0, &bg_init, &[]);
        pass.dispatch_workgroups((bufs.w + 15) / 16, (bufs.h + 15) / 16, 1);
    }

    let mut in_a = true;
    for _ in 0..iterations {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_prop_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.prop_pipeline);
        if in_a {
            pass.set_bind_group(0, &bg_prop_ab, &[]);
        } else {
            pass.set_bind_group(0, &bg_prop_ba, &[]);
        }
        pass.dispatch_workgroups((bufs.w + 15) / 16, (bufs.h + 15) / 16, 1);
        in_a = !in_a;
    }

    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_reset_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.reset_pipeline);
        pass.set_bind_group(0, &bg_reset, &[]);
        pass.dispatch_workgroups((bufs.w + 15) / 16, (bufs.h + 15) / 16, 1);
    }

    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_accum_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.accum_pipeline);
        if in_a {
            pass.set_bind_group(0, &bg_accum_a, &[]);
        } else {
            pass.set_bind_group(0, &bg_accum_b, &[]);
        }
        pass.dispatch_workgroups((bufs.w + 15) / 16, (bufs.h + 15) / 16, 1);
    }

    queue.submit([enc.finish()]);
    in_a
}

pub fn compact_merged_boxes(
    acc_minx: &[u32],
    acc_miny: &[u32],
    acc_maxx: &[u32],
    acc_maxy: &[u32],
    acc_count: &[u32],
    min_count: u32,
) -> Vec<MergedBox> {
    let n = acc_count.len();
    let mut out = Vec::new();
    for i in 0..n {
        let cnt = acc_count[i];
        if cnt < min_count {
            continue;
        }
        let x0 = acc_minx[i];
        let y0 = acc_miny[i];
        if x0 == u32::MAX || y0 == u32::MAX {
            continue;
        }
        let x1 = acc_maxx[i];
        let y1 = acc_maxy[i];
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        out.push(MergedBox {
            x0,
            y0,
            x1,
            y1,
            count: cnt,
        });
    }
    out
}
