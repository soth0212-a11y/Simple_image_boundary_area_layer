use wgpu;

pub struct L2v3Pipelines {
    pub clear_layout: wgpu::BindGroupLayout,
    pub clear_pipeline: wgpu::ComputePipeline,
    pub reduce_layout: wgpu::BindGroupLayout,
    pub reduce_pipeline: wgpu::ComputePipeline,
    pub expand_layout: wgpu::BindGroupLayout,
    pub expand_pipeline: wgpu::ComputePipeline,
    pub emit_layout: wgpu::BindGroupLayout,
    pub emit_pipeline: wgpu::ComputePipeline,
}

pub struct L2v3Buffers {
    pub total_boxes: u32,
    pub bins_x: u32,
    pub bins_y: u32,
    pub bins_count: u32,
    pub band_bins: u32,
    pub groups_count: u32,
    pub group_count: wgpu::Buffer,
    pub group_minx: wgpu::Buffer,
    pub group_miny: wgpu::Buffer,
    pub group_maxx: wgpu::Buffer,
    pub group_maxy: wgpu::Buffer,
    pub group_color: wgpu::Buffer,
    pub out_count: wgpu::Buffer,
    pub out_boxes: wgpu::Buffer,
    pub params_clear: wgpu::Buffer,
    pub params_reduce: wgpu::Buffer,
    pub params_expand: wgpu::Buffer,
    pub params_emit: wgpu::Buffer,
}

pub fn build_l2_v3_pipelines(
    device: &wgpu::Device,
    clear_shader: wgpu::ShaderModule,
    reduce_shader: wgpu::ShaderModule,
    expand_shader: wgpu::ShaderModule,
    emit_shader: wgpu::ShaderModule,
) -> L2v3Pipelines {
    let clear_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v3_clear_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let clear_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v3_clear_pipeline_layout"),
        bind_group_layouts: &[&clear_layout],
        push_constant_ranges: &[],
    });
    let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v3_clear_pipeline"),
        layout: Some(&clear_pipeline_layout),
        module: &clear_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v3_reduce_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let reduce_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v3_reduce_pipeline_layout"),
        bind_group_layouts: &[&reduce_layout],
        push_constant_ranges: &[],
    });
    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v3_reduce_pipeline"),
        layout: Some(&reduce_pipeline_layout),
        module: &reduce_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let expand_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v3_expand_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let expand_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v3_expand_pipeline_layout"),
        bind_group_layouts: &[&expand_layout],
        push_constant_ranges: &[],
    });
    let expand_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v3_expand_pipeline"),
        layout: Some(&expand_pipeline_layout),
        module: &expand_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let emit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v3_emit_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let emit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v3_emit_pipeline_layout"),
        bind_group_layouts: &[&emit_layout],
        push_constant_ranges: &[],
    });
    let emit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v3_emit_pipeline"),
        layout: Some(&emit_pipeline_layout),
        module: &emit_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    L2v3Pipelines {
        clear_layout,
        clear_pipeline,
        reduce_layout,
        reduce_pipeline,
        expand_layout,
        expand_pipeline,
        emit_layout,
        emit_pipeline,
    }
}

fn clamp_u32(v: u32, lo: u32, hi: u32) -> u32 {
    v.max(lo).min(hi)
}

pub fn ensure_l2_v3_buffers(
    device: &wgpu::Device,
    total_boxes: u32,
    bins_x: u32,
    bins_y: u32,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
) -> L2v3Buffers {
    let total = total_boxes.max(1);
    let bins_x = bins_x.max(1);
    let bins_y = bins_y.max(1);
    let bins_count = bins_x.saturating_mul(bins_y).max(1);
    let r_shift = clamp_u32(r_shift, 0, 4);
    let g_shift = clamp_u32(g_shift, 0, 5);
    let b_shift = clamp_u32(b_shift, 0, 4);
    let bits_r = 5u32.saturating_sub(r_shift);
    let bits_g = 6u32.saturating_sub(g_shift);
    let bits_b = 5u32.saturating_sub(b_shift);
    let bits_sum = (bits_r + bits_g + bits_b).min(24);
    let band_bins = (1u32 << bits_sum).max(1);
    let groups_count = band_bins.saturating_mul(bins_count).max(1);

    let group_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_group_count"),
        size: (groups_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group_minx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_group_minx"),
        size: (groups_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group_miny = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_group_miny"),
        size: (groups_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group_maxx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_group_maxx"),
        size: (groups_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group_maxy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_group_maxy"),
        size: (groups_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group_color = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_group_color"),
        size: (groups_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let out_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_out_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_out_boxes"),
        size: (total as u64) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_clear = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_params_clear"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_reduce = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_params_reduce"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_expand = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_params_expand"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_emit = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v3_params_emit"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    L2v3Buffers {
        total_boxes: total,
        bins_x,
        bins_y,
        bins_count,
        band_bins,
        groups_count,
        group_count,
        group_minx,
        group_miny,
        group_maxx,
        group_maxy,
        group_color,
        out_count,
        out_boxes,
        params_clear,
        params_reduce,
        params_expand,
        params_emit,
    }
}
