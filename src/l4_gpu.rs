use wgpu;

pub struct L4ExpandPipelines {
    pub expand_layout: wgpu::BindGroupLayout,
    pub clear_bins: wgpu::ComputePipeline,
    pub insert_bins: wgpu::ComputePipeline,
    pub expand: wgpu::ComputePipeline,
}

pub struct L4ExpandBuffers {
    pub n: u32,
    pub bin_count: u32,
    pub params: wgpu::Buffer,
    pub bin_counts: wgpu::Buffer,
    pub bin_items: wgpu::Buffer,
    pub expanded_boxes: wgpu::Buffer,
    pub expanded_flags: wgpu::Buffer,
}

pub fn build_l4_expand_pipelines(device: &wgpu::Device, expand_module: wgpu::ShaderModule) -> L4ExpandPipelines {
    let expand_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("layer4_expand_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let expand_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("layer4_expand_pipeline_layout"),
        bind_group_layouts: &[&expand_layout],
        push_constant_ranges: &[],
    });

    let clear_bins = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("layer4_clear_bins_pipeline"),
        layout: Some(&expand_pipeline_layout),
        module: &expand_module,
        entry_point: Some("clear_bins"),
        compilation_options: Default::default(),
        cache: None,
    });

    let insert_bins = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("layer4_insert_bins_pipeline"),
        layout: Some(&expand_pipeline_layout),
        module: &expand_module,
        entry_point: Some("insert_bins"),
        compilation_options: Default::default(),
        cache: None,
    });

    let expand = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("layer4_expand_pipeline"),
        layout: Some(&expand_pipeline_layout),
        module: &expand_module,
        entry_point: Some("expand_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    L4ExpandPipelines { expand_layout, clear_bins, insert_bins, expand }
}

pub fn ensure_l4_expand_buffers(device: &wgpu::Device, n: u32, bin_count: u32) -> L4ExpandBuffers {
    let n = n.max(1);
    let params_bytes = 16u64;
    let boxes_bytes = (n as u64) * 2 * 4;
    let flags_bytes = (n as u64) * 4;
    let bin_counts_bytes = (bin_count.max(1) as u64) * 4;
    let bin_items_bytes = (bin_count.max(1) as u64) * 64u64 * 4u64;

    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer4_params"),
        size: params_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bin_counts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer4_bin_counts"),
        size: bin_counts_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bin_items = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer4_bin_items"),
        size: bin_items_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let expanded_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer4_expanded_boxes"),
        size: boxes_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let expanded_flags = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer4_expanded_flags"),
        size: flags_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    L4ExpandBuffers { n, bin_count, params, bin_counts, bin_items, expanded_boxes, expanded_flags }
}
