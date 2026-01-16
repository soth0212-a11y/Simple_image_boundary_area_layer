use wgpu;

pub struct L2EdgesPipelines {
    pub boundary_layout: wgpu::BindGroupLayout,
    pub boundary_pipeline: wgpu::ComputePipeline,
    pub label_layout: wgpu::BindGroupLayout,
    pub label_init_pipeline: wgpu::ComputePipeline,
    pub label_prop_pipeline: wgpu::ComputePipeline,
    pub bbox_layout: wgpu::BindGroupLayout,
    pub bbox_clear_pipeline: wgpu::ComputePipeline,
    pub bbox_accum_pipeline: wgpu::ComputePipeline,
    pub bbox_compact_pipeline: wgpu::ComputePipeline,
}

pub struct L2EdgesBuffers {
    pub w: u32,
    pub h: u32,
    pub max_boxes: u32,
    pub params_boundary: wgpu::Buffer,
    pub params_bbox: wgpu::Buffer,
    pub boundary4: wgpu::Buffer,
    pub label_a: wgpu::Buffer,
    pub label_b: wgpu::Buffer,
    pub bbox_minx: wgpu::Buffer,
    pub bbox_miny: wgpu::Buffer,
    pub bbox_maxx: wgpu::Buffer,
    pub bbox_maxy: wgpu::Buffer,
    pub edge_count: wgpu::Buffer,
    pub boxes_out: wgpu::Buffer,
    pub box_count: wgpu::Buffer,
}

pub fn default_iters(w: u32, h: u32, config_iter: u32) -> u32 {
    let max_dim = w.max(h);
    let base = if config_iter == 0 { max_dim } else { config_iter };
    base.min(max_dim).min(256).max(1)
}

pub fn build_l2_edges_pipelines(
    device: &wgpu::Device,
    boundary_shader: wgpu::ShaderModule,
    label_shader: wgpu::ShaderModule,
    bbox_shader: wgpu::ShaderModule,
) -> L2EdgesPipelines {
    let boundary_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_boundary_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let boundary_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_boundary_pipeline_layout"),
        bind_group_layouts: &[&boundary_layout],
        push_constant_ranges: &[],
    });
    let boundary_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_boundary_pipeline"),
        layout: Some(&boundary_pipeline_layout),
        module: &boundary_shader,
        entry_point: Some("l2_boundary"),
        compilation_options: Default::default(),
        cache: None,
    });

    let label_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_label_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let label_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_label_pipeline_layout"),
        bind_group_layouts: &[&label_layout],
        push_constant_ranges: &[],
    });
    let label_init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_label_init_pipeline"),
        layout: Some(&label_pipeline_layout),
        module: &label_shader,
        entry_point: Some("l2_label_init"),
        compilation_options: Default::default(),
        cache: None,
    });
    let label_prop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_label_prop_pipeline"),
        layout: Some(&label_pipeline_layout),
        module: &label_shader,
        entry_point: Some("l2_label_propagate"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bbox_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_bbox_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let bbox_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l2_bbox_pipeline_layout"),
        bind_group_layouts: &[&bbox_layout],
        push_constant_ranges: &[],
    });
    let bbox_clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_bbox_clear_pipeline"),
        layout: Some(&bbox_pipeline_layout),
        module: &bbox_shader,
        entry_point: Some("l2_bbox_clear"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bbox_accum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_bbox_accum_pipeline"),
        layout: Some(&bbox_pipeline_layout),
        module: &bbox_shader,
        entry_point: Some("l2_bbox_accum"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bbox_compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_bbox_compact_pipeline"),
        layout: Some(&bbox_pipeline_layout),
        module: &bbox_shader,
        entry_point: Some("l2_bbox_compact"),
        compilation_options: Default::default(),
        cache: None,
    });

    L2EdgesPipelines {
        boundary_layout,
        boundary_pipeline,
        label_layout,
        label_init_pipeline,
        label_prop_pipeline,
        bbox_layout,
        bbox_clear_pipeline,
        bbox_accum_pipeline,
        bbox_compact_pipeline,
    }
}

pub fn ensure_l2_edges_buffers(device: &wgpu::Device, w: u32, h: u32, max_boxes: u32) -> L2EdgesBuffers {
    let params_boundary = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_boundary_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_bbox = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_params"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let cell_count = (w * h).max(1) as u64;
    let boundary4 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_boundary4"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let label_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_edge_label_a"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let label_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_edge_label_b"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bbox_minx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_minx"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox_miny = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_miny"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox_maxx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_maxx"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bbox_maxy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_bbox_maxy"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let edge_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_edge_count"),
        size: cell_count * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let box_capacity = max_boxes.max(1) as u64;
    let boxes_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_boxes_out"),
        size: box_capacity * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let box_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_box_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    L2EdgesBuffers {
        w,
        h,
        max_boxes,
        params_boundary,
        params_bbox,
        boundary4,
        label_a,
        label_b,
        bbox_minx,
        bbox_miny,
        bbox_maxx,
        bbox_maxy,
        edge_count,
        boxes_out,
        box_count,
    }
}

pub fn dispatch_l2_edges(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &L2EdgesPipelines,
    bufs: &L2EdgesBuffers,
    plane_id: &wgpu::Buffer,
    edge4: &wgpu::Buffer,
    iters: u32,
    edge_cell_th: u32,
    area_th: u32,
) -> bool {
    let params_boundary = [bufs.w, bufs.h, 0u32, 0u32];
    queue.write_buffer(&bufs.params_boundary, 0, bytemuck::cast_slice(&params_boundary));
    let params_bbox = [bufs.w, bufs.h, edge_cell_th, area_th, bufs.max_boxes, 0u32, 0u32, 0u32];
    queue.write_buffer(&bufs.params_bbox, 0, bytemuck::cast_slice(&params_bbox));

    let bg_boundary = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l2_boundary_bg"),
        layout: &pipelines.boundary_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: plane_id.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: edge4.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.boundary4.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.params_boundary.as_entire_binding() },
        ],
    });

    let bg_label_init = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l2_label_init_bg"),
        layout: &pipelines.label_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bufs.boundary4.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.params_boundary.as_entire_binding() },
        ],
    });

    let bg_label_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l2_label_ab_bg"),
        layout: &pipelines.label_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bufs.boundary4.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.params_boundary.as_entire_binding() },
        ],
    });
    let bg_label_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l2_label_ba_bg"),
        layout: &pipelines.label_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: bufs.boundary4.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs.label_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.label_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.params_boundary.as_entire_binding() },
        ],
    });

    let dispatch_x = (bufs.w + 15) / 16;
    let dispatch_y = (bufs.h + 15) / 16;

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_edges_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_boundary_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.boundary_pipeline);
        pass.set_bind_group(0, &bg_boundary, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_label_init_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.label_init_pipeline);
        pass.set_bind_group(0, &bg_label_init, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    let mut in_a = true;
    for _ in 0..iters {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_label_prop_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.label_prop_pipeline);
        if in_a {
            pass.set_bind_group(0, &bg_label_ab, &[]);
        } else {
            pass.set_bind_group(0, &bg_label_ba, &[]);
        }
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        in_a = !in_a;
    }

    let label_buf = if in_a { &bufs.label_a } else { &bufs.label_b };
    let bg_bbox = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l2_bbox_bg"),
        layout: &pipelines.bbox_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: label_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs.bbox_minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.bbox_miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.bbox_maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.bbox_maxy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.edge_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.boxes_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.box_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.params_bbox.as_entire_binding() },
        ],
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_bbox_clear_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.bbox_clear_pipeline);
        pass.set_bind_group(0, &bg_bbox, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_bbox_accum_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.bbox_accum_pipeline);
        pass.set_bind_group(0, &bg_bbox, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_bbox_compact_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.bbox_compact_pipeline);
        pass.set_bind_group(0, &bg_bbox, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    queue.submit([enc.finish()]);

    in_a
}
