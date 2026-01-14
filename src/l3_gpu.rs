use wgpu;

pub struct L3GpuPipelines {
    pub pass1_layout: wgpu::BindGroupLayout,
    pub pass2a_layout: wgpu::BindGroupLayout,
    pub pass2b_layout: wgpu::BindGroupLayout,
    pub pass1: wgpu::ComputePipeline,
    pub pass2a: wgpu::ComputePipeline,
    pub pass2b: wgpu::ComputePipeline,
}

pub struct L3GpuBuffers {
    pub aw: u32,
    pub ah: u32,
    pub bw: u32,
    pub bh: u32,
    pub bw2: u32,
    pub bh2: u32,
    pub anchor_bbox: wgpu::Buffer,
    pub anchor_score: wgpu::Buffer,
    pub anchor_meta: wgpu::Buffer,
    pub stage1_boxes: wgpu::Buffer,
    pub stage1_scores: wgpu::Buffer,
    pub stage1_valid: wgpu::Buffer,
    pub block_boxes: wgpu::Buffer,
    pub block_scores: wgpu::Buffer,
    pub block_valid: wgpu::Buffer,
}

pub fn build_l3_pipelines(
    device: &wgpu::Device,
    pass1_module: wgpu::ShaderModule,
    pass2a_module: wgpu::ShaderModule,
    pass2b_module: wgpu::ShaderModule,
) -> L3GpuPipelines {
    let pass1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l3_pass1_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pass1_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l3_pass1_pipeline_layout"),
        bind_group_layouts: &[&pass1_layout],
        push_constant_ranges: &[],
    });

    let pass1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l3_pass1_pipeline"),
        layout: Some(&pass1_pipeline_layout),
        module: &pass1_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pass2a_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l3_pass2a_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pass2a_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l3_pass2a_pipeline_layout"),
        bind_group_layouts: &[&pass2a_layout],
        push_constant_ranges: &[],
    });

    let pass2a = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l3_pass2a_pipeline"),
        layout: Some(&pass2a_pipeline_layout),
        module: &pass2a_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pass2b_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l3_pass2b_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pass2b_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l3_pass2b_pipeline_layout"),
        bind_group_layouts: &[&pass2b_layout],
        push_constant_ranges: &[],
    });

    let pass2b = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l3_pass2b_pipeline"),
        layout: Some(&pass2b_pipeline_layout),
        module: &pass2b_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    L3GpuPipelines { pass1_layout, pass2a_layout, pass2b_layout, pass1, pass2a, pass2b }
}

pub fn ensure_l3_buffers(device: &wgpu::Device, aw: u32, ah: u32, bw: u32, bh: u32, bw2: u32, bh2: u32) -> L3GpuBuffers {
    let anchor_count = aw.saturating_mul(ah) as u64;
    let stage1_count = bw.saturating_mul(bh) as u64;
    let block_count = bw2.saturating_mul(bh2) as u64;
    let anchor_bbox_bytes = anchor_count * 2 * 4;
    let anchor_score_bytes = anchor_count * 4;
    let stage1_boxes_bytes = stage1_count * 4 * 2 * 4;
    let stage1_scores_bytes = stage1_count * 4 * 4;
    let block_boxes_bytes = block_count * 16 * 2 * 4;
    let block_scores_bytes = block_count * 16 * 4;

    let anchor_bbox = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_anchor_bbox"),
        size: anchor_bbox_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let anchor_score = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_anchor_score"),
        size: anchor_score_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let anchor_meta = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_anchor_meta"),
        size: anchor_score_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let stage1_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_stage1_boxes"),
        size: stage1_boxes_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stage1_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_stage1_scores"),
        size: stage1_scores_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stage1_valid = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_stage1_valid"),
        size: stage1_scores_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let block_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_block_boxes"),
        size: block_boxes_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let block_scores = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_block_scores"),
        size: block_scores_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let block_valid = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_block_valid"),
        size: block_scores_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    L3GpuBuffers {
        aw,
        ah,
        bw,
        bh,
        bw2,
        bh2,
        anchor_bbox,
        anchor_score,
        anchor_meta,
        stage1_boxes,
        stage1_scores,
        stage1_valid,
        block_boxes,
        block_scores,
        block_valid,
    }
}

pub fn dispatch_l3(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &L3GpuPipelines,
    bufs: &L3GpuBuffers,
    input_info: &wgpu::Buffer,
    pooled_mask: &wgpu::Buffer,
    aw: u32,
    ah: u32,
    bw: u32,
    bh: u32,
    bw2: u32,
    bh2: u32,
) {
    if aw == 0 || ah == 0 || bw == 0 || bh == 0 || bw2 == 0 || bh2 == 0 {
        return;
    }

    let pass1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l3_pass1_bg"),
        layout: &pipelines.pass1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_info.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pooled_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.anchor_bbox.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.anchor_score.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.anchor_meta.as_entire_binding() },
        ],
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l3_pass1"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.pass1);
        pass.set_bind_group(0, &pass1_bg, &[]);
        pass.dispatch_workgroups((aw + 7) / 8, (ah + 7) / 8, 1);
    }

    let pass2a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l3_pass2a_bg"),
        layout: &pipelines.pass2a_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_info.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs.anchor_bbox.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.anchor_score.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.anchor_meta.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.stage1_boxes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.stage1_scores.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.stage1_valid.as_entire_binding() },
        ],
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l3_pass2a"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.pass2a);
        pass.set_bind_group(0, &pass2a_bg, &[]);
        pass.dispatch_workgroups((bw + 7) / 8, (bh + 7) / 8, 1);
    }

    let pass2b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l3_pass2b_bg"),
        layout: &pipelines.pass2b_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_info.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs.stage1_boxes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.stage1_scores.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.stage1_valid.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.block_boxes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.block_scores.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.block_valid.as_entire_binding() },
        ],
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("l3_pass2b"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.pass2b);
        pass.set_bind_group(0, &pass2b_bg, &[]);
        pass.dispatch_workgroups((bw2 + 7) / 8, (bh2 + 7) / 8, 1);
    }
}
