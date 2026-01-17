use wgpu;

pub struct L2v2Pipelines {
    pub clear_layout: wgpu::BindGroupLayout,
    pub clear_pipeline: wgpu::ComputePipeline,
    pub bin_count_layout: wgpu::BindGroupLayout,
    pub bin_count_pipeline: wgpu::ComputePipeline,
    pub scan_layout: wgpu::BindGroupLayout,
    pub scan_pipeline: wgpu::ComputePipeline,
    pub bin_fill_layout: wgpu::BindGroupLayout,
    pub bin_fill_pipeline: wgpu::ComputePipeline,
    pub label_init_layout: wgpu::BindGroupLayout,
    pub label_init_pipeline: wgpu::ComputePipeline,
    pub label_prop_layout: wgpu::BindGroupLayout,
    pub label_prop_pipeline: wgpu::ComputePipeline,
    pub label_compress_layout: wgpu::BindGroupLayout,
    pub label_compress_pipeline: wgpu::ComputePipeline,
    pub reduce_layout: wgpu::BindGroupLayout,
    pub reduce_pipeline: wgpu::ComputePipeline,
    pub emit_layout: wgpu::BindGroupLayout,
    pub emit_pipeline: wgpu::ComputePipeline,
}

pub struct L2v2Buffers {
    pub total_boxes: u32,
    pub bins_x: u32,
    pub bins_y: u32,
    pub bins_count: u32,
    pub bin_count: wgpu::Buffer,
    pub bin_offset: wgpu::Buffer,
    pub bin_cursor: wgpu::Buffer,
    pub bin_items: wgpu::Buffer,
    pub labels: wgpu::Buffer,
    pub comp_minx: wgpu::Buffer,
    pub comp_miny: wgpu::Buffer,
    pub comp_maxx: wgpu::Buffer,
    pub comp_maxy: wgpu::Buffer,
    pub out_count: wgpu::Buffer,
    pub out_boxes: wgpu::Buffer,
    pub params_clear: wgpu::Buffer,
    pub params_bin_count: wgpu::Buffer,
    pub params_scan: wgpu::Buffer,
    pub params_bin_fill: wgpu::Buffer,
    pub params_label_init: wgpu::Buffer,
    pub params_label_prop: wgpu::Buffer,
    pub params_label_compress: wgpu::Buffer,
    pub params_reduce: wgpu::Buffer,
    pub params_emit: wgpu::Buffer,
}

pub fn build_l2_v2_pipelines(
    device: &wgpu::Device,
    clear_shader: wgpu::ShaderModule,
    bin_count_shader: wgpu::ShaderModule,
    scan_shader: wgpu::ShaderModule,
    bin_fill_shader: wgpu::ShaderModule,
    label_init_shader: wgpu::ShaderModule,
    label_prop_shader: wgpu::ShaderModule,
    label_compress_shader: wgpu::ShaderModule,
    reduce_shader: wgpu::ShaderModule,
    emit_shader: wgpu::ShaderModule,
) -> L2v2Pipelines {
    let clear_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_clear_layout"),
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
        label: Some("l2_v2_clear_pipeline_layout"),
        bind_group_layouts: &[&clear_layout],
        push_constant_ranges: &[],
    });
    let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_clear_pipeline"),
        layout: Some(&clear_pipeline_layout),
        module: &clear_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bin_count_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_bin_count_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let bin_count_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v2_bin_count_pipeline_layout"),
        bind_group_layouts: &[&bin_count_layout],
        push_constant_ranges: &[],
    });
    let bin_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_bin_count_pipeline"),
        layout: Some(&bin_count_pipeline_layout),
        module: &bin_count_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let scan_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_scan_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let scan_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v2_scan_pipeline_layout"),
        bind_group_layouts: &[&scan_layout],
        push_constant_ranges: &[],
    });
    let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_scan_pipeline"),
        layout: Some(&scan_pipeline_layout),
        module: &scan_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bin_fill_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_bin_fill_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let bin_fill_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v2_bin_fill_pipeline_layout"),
        bind_group_layouts: &[&bin_fill_layout],
        push_constant_ranges: &[],
    });
    let bin_fill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_bin_fill_pipeline"),
        layout: Some(&bin_fill_pipeline_layout),
        module: &bin_fill_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let label_init_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_label_init_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let label_init_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v2_label_init_pipeline_layout"),
        bind_group_layouts: &[&label_init_layout],
        push_constant_ranges: &[],
    });
    let label_init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_label_init_pipeline"),
        layout: Some(&label_init_pipeline_layout),
        module: &label_init_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let label_prop_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_label_prop_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let label_prop_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v2_label_prop_pipeline_layout"),
        bind_group_layouts: &[&label_prop_layout],
        push_constant_ranges: &[],
    });
    let label_prop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_label_prop_pipeline"),
        layout: Some(&label_prop_pipeline_layout),
        module: &label_prop_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let label_compress_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_label_compress_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let label_compress_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v2_label_compress_pipeline_layout"),
        bind_group_layouts: &[&label_compress_layout],
        push_constant_ranges: &[],
    });
    let label_compress_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_label_compress_pipeline"),
        layout: Some(&label_compress_pipeline_layout),
        module: &label_compress_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_reduce_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let reduce_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_v2_reduce_pipeline_layout"),
        bind_group_layouts: &[&reduce_layout],
        push_constant_ranges: &[],
    });
    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_reduce_pipeline"),
        layout: Some(&reduce_pipeline_layout),
        module: &reduce_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let emit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_v2_emit_layout"),
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
        label: Some("l2_v2_emit_pipeline_layout"),
        bind_group_layouts: &[&emit_layout],
        push_constant_ranges: &[],
    });
    let emit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_v2_emit_pipeline"),
        layout: Some(&emit_pipeline_layout),
        module: &emit_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    L2v2Pipelines {
        clear_layout,
        clear_pipeline,
        bin_count_layout,
        bin_count_pipeline,
        scan_layout,
        scan_pipeline,
        bin_fill_layout,
        bin_fill_pipeline,
        label_init_layout,
        label_init_pipeline,
        label_prop_layout,
        label_prop_pipeline,
        label_compress_layout,
        label_compress_pipeline,
        reduce_layout,
        reduce_pipeline,
        emit_layout,
        emit_pipeline,
    }
}

pub fn ensure_l2_v2_buffers(
    device: &wgpu::Device,
    total_boxes: u32,
    bins_x: u32,
    bins_y: u32,
) -> L2v2Buffers {
    let bins_count = bins_x.saturating_mul(bins_y).max(1);
    let total = total_boxes.max(1);
    let bin_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_bin_count"),
        size: (bins_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bin_offset = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_bin_offset"),
        size: ((bins_count + 1) as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bin_cursor = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_bin_cursor"),
        size: (bins_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bin_items = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_bin_items"),
        size: (total as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let labels = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_labels"),
        size: (total as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let comp_minx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_comp_minx"),
        size: (total as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let comp_miny = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_comp_miny"),
        size: (total as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let comp_maxx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_comp_maxx"),
        size: (total as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let comp_maxy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_comp_maxy"),
        size: (total as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_out_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_out_boxes"),
        size: (total as u64) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_clear = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_clear"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_bin_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_bin_count"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_scan = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_scan"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_bin_fill = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_bin_fill"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_label_init = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_label_init"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_label_prop = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_label_prop"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_label_compress = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_label_compress"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_reduce = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_reduce"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_emit = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_v2_params_emit"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    L2v2Buffers {
        total_boxes: total_boxes.max(1),
        bins_x,
        bins_y,
        bins_count,
        bin_count,
        bin_offset,
        bin_cursor,
        bin_items,
        labels,
        comp_minx,
        comp_miny,
        comp_maxx,
        comp_maxy,
        out_count,
        out_boxes,
        params_clear,
        params_bin_count,
        params_scan,
        params_bin_fill,
        params_label_init,
        params_label_prop,
        params_label_compress,
        params_reduce,
        params_emit,
    }
}
