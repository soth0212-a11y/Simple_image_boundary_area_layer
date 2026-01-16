use wgpu;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Segment {
    pub x0: u32,
    pub x1: u32,
    pub y_color: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OutBox {
    pub x0y0: u32,
    pub x1y1: u32,
    pub color565: u32,
    pub flags: u32,
}

pub struct RleCclPipelines {
    pub count_layout: wgpu::BindGroupLayout,
    pub count_pipeline: wgpu::ComputePipeline,
    pub emit_layout: wgpu::BindGroupLayout,
    pub emit_pipeline: wgpu::ComputePipeline,
    pub init_layout: wgpu::BindGroupLayout,
    pub init_pipeline: wgpu::ComputePipeline,
    pub union_layout: wgpu::BindGroupLayout,
    pub union_pipeline: wgpu::ComputePipeline,
    pub reduce_layout: wgpu::BindGroupLayout,
    pub reduce_pipeline: wgpu::ComputePipeline,
    pub emit_boxes_layout: wgpu::BindGroupLayout,
    pub emit_boxes_pipeline: wgpu::ComputePipeline,
}

pub struct RleCclBuffers {
    pub w: u32,
    pub h: u32,
    pub total_segments: u32,
    pub max_out: u32,
    pub row_counts: wgpu::Buffer,
    pub row_offsets: wgpu::Buffer,
    pub segments: wgpu::Buffer,
    pub parent: wgpu::Buffer,
    pub bbox_minx: wgpu::Buffer,
    pub bbox_miny: wgpu::Buffer,
    pub bbox_maxx: wgpu::Buffer,
    pub bbox_maxy: wgpu::Buffer,
    pub out_count: wgpu::Buffer,
    pub out_boxes: wgpu::Buffer,
    pub params_count: wgpu::Buffer,
    pub params_emit: wgpu::Buffer,
    pub params_init: wgpu::Buffer,
    pub params_union: wgpu::Buffer,
    pub params_reduce: wgpu::Buffer,
    pub params_emit_boxes: wgpu::Buffer,
}

pub fn build_rle_ccl_pipelines(
    device: &wgpu::Device,
    count_shader: wgpu::ShaderModule,
    emit_shader: wgpu::ShaderModule,
    init_shader: wgpu::ShaderModule,
    union_shader: wgpu::ShaderModule,
    reduce_shader: wgpu::ShaderModule,
    emit_boxes_shader: wgpu::ShaderModule,
) -> RleCclPipelines {
    let count_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l1_rle_count_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let count_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l1_rle_count_pipeline_layout"),
        bind_group_layouts: &[&count_layout],
        push_constant_ranges: &[],
    });
    let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l1_rle_count_pipeline"),
        layout: Some(&count_pipeline_layout),
        module: &count_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let emit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l1_rle_emit_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let emit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l1_rle_emit_pipeline_layout"),
        bind_group_layouts: &[&emit_layout],
        push_constant_ranges: &[],
    });
    let emit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l1_rle_emit_pipeline"),
        layout: Some(&emit_pipeline_layout),
        module: &emit_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let init_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_ccl_init_layout"),
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
    let init_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_ccl_init_pipeline_layout"),
        bind_group_layouts: &[&init_layout],
        push_constant_ranges: &[],
    });
    let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_ccl_init_pipeline"),
        layout: Some(&init_pipeline_layout),
        module: &init_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let union_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_ccl_union_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let union_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_ccl_union_pipeline_layout"),
        bind_group_layouts: &[&union_layout],
        push_constant_ranges: &[],
    });
    let union_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_ccl_union_pipeline"),
        layout: Some(&union_pipeline_layout),
        module: &union_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_ccl_reduce_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let reduce_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_ccl_reduce_pipeline_layout"),
        bind_group_layouts: &[&reduce_layout],
        push_constant_ranges: &[],
    });
    let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_ccl_reduce_pipeline"),
        layout: Some(&reduce_pipeline_layout),
        module: &reduce_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let emit_boxes_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_ccl_emit_boxes_layout"),
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
    let emit_boxes_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_ccl_emit_boxes_pipeline_layout"),
        bind_group_layouts: &[&emit_boxes_layout],
        push_constant_ranges: &[],
    });
    let emit_boxes_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_ccl_emit_boxes_pipeline"),
        layout: Some(&emit_boxes_pipeline_layout),
        module: &emit_boxes_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    RleCclPipelines {
        count_layout,
        count_pipeline,
        emit_layout,
        emit_pipeline,
        init_layout,
        init_pipeline,
        union_layout,
        union_pipeline,
        reduce_layout,
        reduce_pipeline,
        emit_boxes_layout,
        emit_boxes_pipeline,
    }
}

pub fn ensure_rle_ccl_buffers(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    total_segments: u32,
    max_out: u32,
) -> RleCclBuffers {
    let row_counts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_row_counts"),
        size: (h.max(1) as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let row_offsets = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_row_offsets"),
        size: ((h + 1).max(1) as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let seg_count = total_segments.max(1) as u64;
    let segments = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_segments"),
        size: seg_count * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let parent = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_parent"),
        size: seg_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bbox_minx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_minx"),
        size: seg_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox_miny = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_miny"),
        size: seg_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox_maxx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_maxx"),
        size: seg_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox_maxy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_maxy"),
        size: seg_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let out_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_out_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_out_boxes"),
        size: (max_out.max(1) as u64) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_count_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_emit = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l1_emit_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_init = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_init_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_union = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_union_params"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_reduce = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_reduce_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_emit_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_emit_boxes_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    RleCclBuffers {
        w,
        h,
        total_segments,
        max_out,
        row_counts,
        row_offsets,
        segments,
        parent,
        bbox_minx,
        bbox_miny,
        bbox_maxx,
        bbox_maxy,
        out_count,
        out_boxes,
        params_count,
        params_emit,
        params_init,
        params_union,
        params_reduce,
        params_emit_boxes,
    }
}
