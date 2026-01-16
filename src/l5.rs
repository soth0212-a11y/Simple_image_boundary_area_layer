use wgpu;

pub const L5_WIN: u32 = 2;
pub const L5_STR: u32 = 1;
pub const L5_MARGIN: u32 = 8;

pub struct L5Pipelines {
    pub layout: wgpu::BindGroupLayout,
    pub score_pipeline: wgpu::ComputePipeline,
    pub thresh_pipeline: wgpu::ComputePipeline,
    pub roi_pipeline: wgpu::ComputePipeline,
    pub init_pipeline: wgpu::ComputePipeline,
    pub prop_pipeline: wgpu::ComputePipeline,
    pub reset_pipeline: wgpu::ComputePipeline,
    pub accum_pipeline: wgpu::ComputePipeline,
}

pub struct L5Buffers {
    pub w: u32,
    pub h: u32,
    pub tile_w: u32,
    pub tile_h: u32,
    pub params0: wgpu::Buffer,
    pub params1: wgpu::Buffer,
    pub score_map: wgpu::Buffer,
    pub tile_keep: wgpu::Buffer,
    pub roi_mask: wgpu::Buffer,
    pub label_a: wgpu::Buffer,
    pub label_b: wgpu::Buffer,
    pub acc_minx: wgpu::Buffer,
    pub acc_miny: wgpu::Buffer,
    pub acc_maxx: wgpu::Buffer,
    pub acc_maxy: wgpu::Buffer,
    pub acc_count: wgpu::Buffer,
}

pub fn tile_dims(w: u32, h: u32) -> (u32, u32) {
    let tile_w = if w <= L5_WIN {
        1
    } else {
        ((w - L5_WIN + L5_STR - 1) / L5_STR) + 1
    };
    let tile_h = if h <= L5_WIN {
        1
    } else {
        ((h - L5_WIN + L5_STR - 1) / L5_STR) + 1
    };
    (tile_w.max(1), tile_h.max(1))
}

pub fn default_threshold() -> u32 {
    12
}

pub fn default_iters(w: u32, h: u32) -> u32 {
    let m = w.max(h);
    m.min(128).max(1)
}

pub fn build_l5_pipelines(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> L5Pipelines {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l5_threshold_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 11, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 12, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 13, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l5_threshold_pipeline_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let score_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_score_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_score_map"),
        compilation_options: Default::default(),
        cache: None,
    });
    let thresh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_threshold_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_threshold_tiles"),
        compilation_options: Default::default(),
        cache: None,
    });
    let roi_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_roi_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_upsample_roi_mask"),
        compilation_options: Default::default(),
        cache: None,
    });
    let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_init_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_init_labels"),
        compilation_options: Default::default(),
        cache: None,
    });
    let prop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_prop_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_propagate_labels"),
        compilation_options: Default::default(),
        cache: None,
    });
    let reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_reset_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_reset_accumulators"),
        compilation_options: Default::default(),
        cache: None,
    });
    let accum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l5_accum_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("l5_accumulate_bboxes"),
        compilation_options: Default::default(),
        cache: None,
    });

    L5Pipelines {
        layout,
        score_pipeline,
        thresh_pipeline,
        roi_pipeline,
        init_pipeline,
        prop_pipeline,
        reset_pipeline,
        accum_pipeline,
    }
}

pub fn ensure_l5_buffers(device: &wgpu::Device, w: u32, h: u32) -> L5Buffers {
    let (tile_w, tile_h) = tile_dims(w, h);
    let params0 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_params0"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params1 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_params1"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let tile_count = (tile_w * tile_h).max(1) as u64;
    let score_map = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_score_map"),
        size: tile_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let tile_keep = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_tile_keep"),
        size: tile_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cell_count = (w * h).max(1) as u64;
    let roi_mask = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_roi_mask"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let label_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_label_a"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let label_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_label_b"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let acc_minx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_minx"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let acc_miny = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_miny"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let acc_maxx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_maxx"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let acc_maxy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_maxy"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let acc_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l5_acc_count"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    L5Buffers {
        w,
        h,
        tile_w,
        tile_h,
        params0,
        params1,
        score_map,
        tile_keep,
        roi_mask,
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
    s_active: &wgpu::Buffer,
    conn8: &wgpu::Buffer,
    threshold: u32,
    iters: u32,
) -> bool {
    let params0 = [bufs.w, bufs.h, bufs.tile_w, bufs.tile_h];
    let params1 = [threshold, iters, 0u32, 0u32];
    queue.write_buffer(&bufs.params0, 0, bytemuck::cast_slice(&params0));
    queue.write_buffer(&bufs.params1, 0, bytemuck::cast_slice(&params1));

    let bg_base = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_bg_base"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: conn8.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.score_map.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.tile_keep.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.roi_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.acc_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.acc_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs.acc_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: bufs.acc_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: bufs.acc_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: bufs.params0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: bufs.params1.as_entire_binding() },
        ],
    });

    let bg_prop_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_bg_prop_ab"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: conn8.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.score_map.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.tile_keep.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.roi_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.acc_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.acc_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs.acc_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: bufs.acc_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: bufs.acc_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: bufs.params0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: bufs.params1.as_entire_binding() },
        ],
    });
    let bg_prop_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_bg_prop_ba"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: conn8.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.score_map.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.tile_keep.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.roi_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.acc_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.acc_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs.acc_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: bufs.acc_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: bufs.acc_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: bufs.params0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: bufs.params1.as_entire_binding() },
        ],
    });
    let bg_accum_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_bg_accum_a"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: conn8.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.score_map.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.tile_keep.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.roi_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.acc_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.acc_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs.acc_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: bufs.acc_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: bufs.acc_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: bufs.params0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: bufs.params1.as_entire_binding() },
        ],
    });
    let bg_accum_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l5_bg_accum_b"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: conn8.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.score_map.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.tile_keep.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.roi_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.acc_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.acc_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs.acc_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: bufs.acc_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: bufs.acc_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: bufs.params0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: bufs.params1.as_entire_binding() },
        ],
    });

    let tile_dispatch_x = (bufs.tile_w + 7) / 8;
    let tile_dispatch_y = (bufs.tile_h + 7) / 8;
    let cell_dispatch_x = (bufs.w + 15) / 16;
    let cell_dispatch_y = (bufs.h + 15) / 16;

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l5_threshold_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_score_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.score_pipeline);
        pass.set_bind_group(0, &bg_base, &[]);
        pass.dispatch_workgroups(tile_dispatch_x, tile_dispatch_y, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_thresh_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.thresh_pipeline);
        pass.set_bind_group(0, &bg_base, &[]);
        pass.dispatch_workgroups(tile_dispatch_x, tile_dispatch_y, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_roi_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.roi_pipeline);
        pass.set_bind_group(0, &bg_base, &[]);
        pass.dispatch_workgroups(cell_dispatch_x, cell_dispatch_y, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_init_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.init_pipeline);
        pass.set_bind_group(0, &bg_base, &[]);
        pass.dispatch_workgroups(cell_dispatch_x, cell_dispatch_y, 1);
    }

    let mut in_a = true;
    for _ in 0..iters {
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
        pass.dispatch_workgroups(cell_dispatch_x, cell_dispatch_y, 1);
        in_a = !in_a;
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l5_reset_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.reset_pipeline);
        pass.set_bind_group(0, &bg_base, &[]);
        pass.dispatch_workgroups(cell_dispatch_x, cell_dispatch_y, 1);
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
        pass.dispatch_workgroups(cell_dispatch_x, cell_dispatch_y, 1);
    }

    queue.submit([enc.finish()]);
    in_a
}
