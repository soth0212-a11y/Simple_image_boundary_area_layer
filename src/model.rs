use std::path::Path;
use std::time::{Duration, Instant};
use wgpu;
use wgpu::util::DeviceExt;

use crate::config;
use crate::preprocessing;
use crate::rle_ccl_gpu;
use crate::visualization;

pub struct Layer0Outputs {
    pub packed: wgpu::Buffer,
    pub cell_rgb: wgpu::Buffer,
    pub edge4: wgpu::Buffer,
    pub s_active: wgpu::Buffer,
    pub color565: wgpu::Buffer,
}

pub struct layer_static_values {
    pub bindgroup_layout: wgpu::BindGroupLayout,
    pub compute_pipeline: wgpu::ComputePipeline,
}

fn readback_u32_buffer(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, byte_len: u64) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

fn readback_segments(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, count: u32) -> Vec<rle_ccl_gpu::Segment> {
    if count == 0 {
        return Vec::new();
    }
    let bytes = (count as u64) * 16;
    let raw = readback_u32_buffer(device, queue, src, bytes);
    raw.chunks_exact(4)
        .map(|c| rle_ccl_gpu::Segment { x0: c[0], x1: c[1], y_color: c[2], pad: c[3] })
        .collect()
}

fn readback_out_boxes(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, count: u32) -> Vec<rle_ccl_gpu::OutBox> {
    if count == 0 {
        return Vec::new();
    }
    let bytes = (count as u64) * 16;
    let raw = readback_u32_buffer(device, queue, src, bytes);
    raw.chunks_exact(4)
        .map(|c| rle_ccl_gpu::OutBox { x0y0: c[0], x1y1: c[1], color565: c[2], flags: c[3] })
        .collect()
}

fn prefix_sum(counts: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0);
    for &c in counts {
        let last = *offsets.last().unwrap();
        offsets.push(last + c);
    }
    offsets
}

pub fn layer0_init(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> layer_static_values {
    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l0_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ][..],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bindgroup_layout], push_constant_ranges: &[][..] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    layer_static_values { bindgroup_layout, compute_pipeline }
}

pub fn layer0(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img_view: &wgpu::TextureView,
    img_info: preprocessing::Imginfo,
    static_vals: &layer_static_values,
) -> (Layer0Outputs, [u32; 4]) {
    let info = [img_info.height, img_info.width, img_info.height, img_info.width];
    let grid_w = img_info.width;
    let grid_h = img_info.height;
    let out_w = grid_w;
    let out_h = grid_h;
    let out_size = (out_w * out_h).max(1) as u64 * 8;
    let out_packed = device.create_buffer(&wgpu::BufferDescriptor { label: Some("layer0_out_packed"), size: out_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let cell_size = (out_w * out_h).max(1) as u64 * 4;
    let cell_rgb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer0_cell_rgb"),
        size: cell_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let edge4 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer0_edge4"),
        size: cell_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let s_active = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer0_s_active"),
        size: cell_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let color565 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer0_color565"),
        size: cell_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let cfg = config::get();
    let info_buf_data = [
        img_info.height,
        img_info.width,
        cfg.l0_edge_th_r,
        cfg.l0_edge_th_g,
        cfg.l0_edge_th_b,
        cfg.l0_dir_min_channels,
        cfg.l0_pixel_min_dirs,
    ];
    let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&info_buf_data), usage: wgpu::BufferUsages::STORAGE });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &static_vals.bindgroup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(img_view) },
            wgpu::BindGroupEntry { binding: 1, resource: info_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_packed.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cell_rgb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: edge4.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: color565.as_entire_binding() },
        ][..],
        label: None,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&static_vals.compute_pipeline);
        pass.set_bind_group(0, &bg, &[][..]);
        pass.dispatch_workgroups((grid_w + 15) / 16, (grid_h + 15) / 16, 1);
    }
    queue.submit([enc.finish()]);
    (Layer0Outputs { packed: out_packed, cell_rgb, edge4, s_active, color565 }, info)
}

pub fn model(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img_view: wgpu::TextureView,
    img_info: preprocessing::Imginfo,
    l0_static: &layer_static_values,
    pipelines: &rle_ccl_gpu::RleCclPipelines,
    buffers: &mut Option<rle_ccl_gpu::RleCclBuffers>,
    src_path: &Path,
) {
    let total_start = Instant::now();

    let l0_start = Instant::now();
    let (l0_out, info) = layer0(device, queue, &img_view, img_info, l0_static);
    device.poll(wgpu::PollType::Wait);
    let l0_dur = l0_start.elapsed();

    let width = info[1];
    let height = info[0];
    if config::get().save_layer0 {
        let l0_bytes = (width * height) as u64 * 8;
        let packed = readback_u32_buffer(device, queue, &l0_out.packed, l0_bytes);
        let _ = visualization::save_layer0_packed_rgb(
            src_path,
            &packed,
            width as usize,
            height as usize,
            "l0",
        );
    }

    let cfg = config::get();
    let max_out = cfg.l2_max_out.max(1);
    let mut offsets: Vec<u32> = Vec::new();
    let total_segments = {
        let rebuild = buffers
            .as_ref()
            .map(|b| b.w != width || b.h != height || b.max_out != max_out)
            .unwrap_or(true);
        if rebuild {
            *buffers = Some(rle_ccl_gpu::ensure_rle_ccl_buffers(device, width, height, 1, max_out));
        }
        let bufs = buffers.as_ref().unwrap();
        let params_count = [width, height, 0u32, 0u32];
        queue.write_buffer(&bufs.params_count, 0, bytemuck::cast_slice(&params_count));
        let bg_count = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l1_rle_count_bg"),
            layout: &pipelines.count_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: l0_out.s_active.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: l0_out.color565.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.row_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bufs.params_count.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l1_rle_count_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l1_rle_count_pass"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.count_pipeline);
            pass.set_bind_group(0, &bg_count, &[]);
            pass.dispatch_workgroups(height, 1, 1);
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::Wait);
        let counts = readback_u32_buffer(device, queue, &bufs.row_counts, (height as u64) * 4);
        offsets = prefix_sum(&counts);
        *offsets.last().unwrap_or(&0)
    };

    let rebuild = buffers
        .as_ref()
        .map(|b| b.w != width || b.h != height || b.total_segments != total_segments || b.max_out != max_out)
        .unwrap_or(true);
    if rebuild {
        *buffers = Some(rle_ccl_gpu::ensure_rle_ccl_buffers(device, width, height, total_segments, max_out));
    }
    let bufs = buffers.as_ref().unwrap();
    if !offsets.is_empty() {
        queue.write_buffer(&bufs.row_offsets, 0, bytemuck::cast_slice(&offsets));
    }

    let params_emit = [width, height, 0u32, 0u32];
    queue.write_buffer(&bufs.params_emit, 0, bytemuck::cast_slice(&params_emit));
    let bg_emit = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l1_rle_emit_bg"),
        layout: &pipelines.emit_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: l0_out.s_active.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: l0_out.color565.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.row_offsets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.segments.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.params_emit.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l1_rle_emit_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l1_rle_emit_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.emit_pipeline);
        pass.set_bind_group(0, &bg_emit, &[]);
        pass.dispatch_workgroups(height, 1, 1);
    }
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::Wait);

    if total_segments > 0 {
        let params_init = [total_segments, 0u32, 0u32, 0u32];
        queue.write_buffer(&bufs.params_init, 0, bytemuck::cast_slice(&params_init));
        let bg_init = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_ccl_init_bg"),
            layout: &pipelines.init_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.parent.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.bbox_minx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.bbox_miny.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bufs.bbox_maxx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bufs.bbox_maxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bufs.out_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: bufs.params_init.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_ccl_init_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_ccl_init_pass"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.init_pipeline);
            pass.set_bind_group(0, &bg_init, &[]);
            pass.dispatch_workgroups((total_segments + 255) / 256, 1, 1);
        }
        queue.submit([enc.finish()]);

        let params_union = [width, height, cfg.l2_color_tol, cfg.l2_gap_x, cfg.l2_gap_y, 0u32, 0u32, 0u32];
        queue.write_buffer(&bufs.params_union, 0, bytemuck::cast_slice(&params_union));
        let bg_union = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_ccl_union_bg"),
            layout: &pipelines.union_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.row_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.segments.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.parent.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bufs.params_union.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_ccl_union_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_ccl_union_pass"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.union_pipeline);
            pass.set_bind_group(0, &bg_union, &[]);
            pass.dispatch_workgroups(height, 1, 1);
        }
        queue.submit([enc.finish()]);

        let params_reduce = [total_segments, 0u32, 0u32, 0u32];
        queue.write_buffer(&bufs.params_reduce, 0, bytemuck::cast_slice(&params_reduce));
        let bg_reduce = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_ccl_reduce_bg"),
            layout: &pipelines.reduce_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.segments.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.parent.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.bbox_minx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bufs.bbox_miny.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bufs.bbox_maxx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bufs.bbox_maxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: bufs.params_reduce.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_ccl_reduce_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_ccl_reduce_pass"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.reduce_pipeline);
            pass.set_bind_group(0, &bg_reduce, &[]);
            pass.dispatch_workgroups((total_segments + 255) / 256, 1, 1);
        }
        queue.submit([enc.finish()]);

        let params_emit_boxes = [total_segments, max_out, cfg.l2_min_w, cfg.l2_min_h];
        queue.write_buffer(&bufs.params_emit_boxes, 0, bytemuck::cast_slice(&params_emit_boxes));
        let bg_emit_boxes = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_ccl_emit_boxes_bg"),
            layout: &pipelines.emit_boxes_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.segments.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs.parent.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs.bbox_minx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bufs.bbox_miny.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: bufs.bbox_maxx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bufs.bbox_maxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: bufs.out_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: bufs.out_boxes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: bufs.params_emit_boxes.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_ccl_emit_boxes_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_ccl_emit_boxes_pass"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.emit_boxes_pipeline);
            pass.set_bind_group(0, &bg_emit_boxes, &[]);
            pass.dispatch_workgroups((total_segments + 255) / 256, 1, 1);
        }
        queue.submit([enc.finish()]);
    }

    if cfg.save_l1_segments {
        let segments = readback_segments(device, queue, &bufs.segments, total_segments);
        let _ = visualization::save_l1_segments_overlay(
            src_path,
            &segments,
            width as usize,
            height as usize,
            "l1",
        );
    }

    if cfg.save_l2_boxes {
        let count = readback_u32_buffer(device, queue, &bufs.out_count, 4).get(0).copied().unwrap_or(0);
        let boxes = readback_out_boxes(device, queue, &bufs.out_boxes, count);
        let _ = visualization::save_l2_boxes_on_src(
            src_path,
            &boxes,
            width as usize,
            height as usize,
            "l2",
        );
    }

    if config::get().log_timing {
        visualization::log_timing_layers(
            src_path,
            info,
            l0_dur,
            total_start.elapsed(),
        );
    }
}
