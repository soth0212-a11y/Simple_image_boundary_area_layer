use std::path::Path;
use std::time::{Duration, Instant};
use wgpu;
use wgpu::util::DeviceExt;

use crate::config;
use crate::layer3_cpu;
use crate::l2_v3_gpu;
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
        .map(|c| rle_ccl_gpu::Segment { tl: c[0], br: c[1], color565: c[2], pad: c[3] })
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

fn readback_color565_map(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, count: u32) -> Vec<u32> {
    if count == 0 {
        return Vec::new();
    }
    readback_u32_buffer(device, queue, src, (count as u64) * 4)
}

fn read_l2_boxes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    out_count_buf: &wgpu::Buffer,
    out_boxes_buf: &wgpu::Buffer,
    max_count: u32,
) -> Vec<rle_ccl_gpu::OutBox> {
    let count = readback_u32_buffer(device, queue, out_count_buf, 4)
        .get(0)
        .copied()
        .unwrap_or(0)
        .min(max_count);
    readback_out_boxes(device, queue, out_boxes_buf, count)
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
    l1_pipelines: &rle_ccl_gpu::L1RlePipelines,
    l1_buffers: &mut Option<rle_ccl_gpu::L1RleBuffers>,
    l2_pipelines: &l2_v3_gpu::L2v3Pipelines,
    l2_buffers: &mut Option<l2_v3_gpu::L2v3Buffers>,
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
    let mut offsets: Vec<u32> = Vec::new();
    let l1_start = Instant::now();
    let total_segments = {
        let rebuild = l1_buffers
            .as_ref()
            .map(|b| b.w != width || b.h != height)
            .unwrap_or(true);
        if rebuild {
            *l1_buffers = Some(rle_ccl_gpu::ensure_l1_rle_buffers(device, width, height, 1));
        }
        let bufs = l1_buffers.as_ref().unwrap();
        let params_count = [width, height, 0u32, 0u32];
        queue.write_buffer(&bufs.params_count, 0, bytemuck::cast_slice(&params_count));
        let bg_count = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l1_rle_count_bg"),
            layout: &l1_pipelines.count_layout,
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
            pass.set_pipeline(&l1_pipelines.count_pipeline);
            pass.set_bind_group(0, &bg_count, &[]);
            pass.dispatch_workgroups(height, 1, 1);
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::Wait);
        let counts = readback_u32_buffer(device, queue, &bufs.row_counts, (height as u64) * 4);
        offsets = prefix_sum(&counts);
        *offsets.last().unwrap_or(&0)
    };

    let rebuild = l1_buffers
        .as_ref()
        .map(|b| b.w != width || b.h != height || b.total_segments != total_segments)
        .unwrap_or(true);
    if rebuild {
        *l1_buffers = Some(rle_ccl_gpu::ensure_l1_rle_buffers(device, width, height, total_segments));
    }
    let bufs = l1_buffers.as_ref().unwrap();
    if !offsets.is_empty() {
        queue.write_buffer(&bufs.row_offsets, 0, bytemuck::cast_slice(&offsets));
    }

    let params_emit = [width, height, cfg.l1_color_tol, 0u32];
    queue.write_buffer(&bufs.params_emit, 0, bytemuck::cast_slice(&params_emit));
    let bg_emit = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l1_rle_emit_bg"),
        layout: &l1_pipelines.emit_layout,
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
        pass.set_pipeline(&l1_pipelines.emit_pipeline);
        pass.set_bind_group(0, &bg_emit, &[]);
        pass.dispatch_workgroups(height, 1, 1);
    }
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::Wait);
    let l1_dur = l1_start.elapsed();

    let l2_start = Instant::now();
    let mut l2_pass_times: Vec<(&'static str, Duration)> = Vec::new();
    if total_segments > 0 {
        let bin_size = cfg.l2_bin_size.max(1);
        let bins_x = ((width + bin_size - 1) / bin_size).max(1);
        let bins_y = ((height + bin_size - 1) / bin_size).max(1);
        let r_shift = cfg.l2_band_r_shift;
        let g_shift = cfg.l2_band_g_shift;
        let b_shift = cfg.l2_band_b_shift;

        let clamp_u32 = |v: u32, lo: u32, hi: u32| v.max(lo).min(hi);
        let r_shift_c = clamp_u32(r_shift, 0, 4);
        let g_shift_c = clamp_u32(g_shift, 0, 5);
        let b_shift_c = clamp_u32(b_shift, 0, 4);
        let bits_r = 5u32.saturating_sub(r_shift_c);
        let bits_g = 6u32.saturating_sub(g_shift_c);
        let bits_b = 5u32.saturating_sub(b_shift_c);
        let bits_sum = (bits_r + bits_g + bits_b).min(24);
        let expected_band_bins = (1u32 << bits_sum).max(1);

        let rebuild = l2_buffers
            .as_ref()
            .map(|b| {
                b.total_boxes != total_segments
                    || b.bins_x != bins_x
                    || b.bins_y != bins_y
                    || b.band_bins != expected_band_bins
            })
            .unwrap_or(true);
        if rebuild {
            *l2_buffers = Some(l2_v3_gpu::ensure_l2_v3_buffers(
                device,
                total_segments,
                bins_x,
                bins_y,
                r_shift,
                g_shift,
                b_shift,
            ));
        }
        let l2_bufs = l2_buffers.as_ref().unwrap();

        let pass_start = Instant::now();
        let params_clear = [l2_bufs.groups_count, 0u32, 0u32, 0u32];
        queue.write_buffer(&l2_bufs.params_clear, 0, bytemuck::cast_slice(&params_clear));
        let bg_clear = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_v3_clear_bg"),
            layout: &l2_pipelines.clear_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: l2_bufs.group_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: l2_bufs.group_minx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: l2_bufs.group_miny.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: l2_bufs.group_maxx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: l2_bufs.group_maxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: l2_bufs.group_color.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: l2_bufs.out_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: l2_bufs.params_clear.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_v3_clear_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_v3_clear_pass"), timestamp_writes: None });
            pass.set_pipeline(&l2_pipelines.clear_pipeline);
            pass.set_bind_group(0, &bg_clear, &[]);
            pass.dispatch_workgroups((l2_bufs.groups_count + 255) / 256, 1, 1);
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::Wait);
        l2_pass_times.push(("l2_clear", pass_start.elapsed()));

        let pass_start = Instant::now();
        let params_reduce = [total_segments, bins_x, bins_y, bin_size, r_shift, g_shift, b_shift, 0u32];
        queue.write_buffer(&l2_bufs.params_reduce, 0, bytemuck::cast_slice(&params_reduce));
        let bg_reduce = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_v3_reduce_bg"),
            layout: &l2_pipelines.reduce_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bufs.segments.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: l2_bufs.group_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: l2_bufs.group_minx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: l2_bufs.group_miny.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: l2_bufs.group_maxx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: l2_bufs.group_maxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: l2_bufs.group_color.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: l2_bufs.params_reduce.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_v3_reduce_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_v3_reduce_pass"), timestamp_writes: None });
            pass.set_pipeline(&l2_pipelines.reduce_pipeline);
            pass.set_bind_group(0, &bg_reduce, &[]);
            pass.dispatch_workgroups((total_segments + 255) / 256, 1, 1);
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::Wait);
        l2_pass_times.push(("l2_group_reduce", pass_start.elapsed()));

        let params_expand = [
            l2_bufs.groups_count,
            bins_x,
            bins_y,
            cfg.l2_color_tol,
            r_shift,
            g_shift,
            b_shift,
            0u32,
        ];
        queue.write_buffer(&l2_bufs.params_expand, 0, bytemuck::cast_slice(&params_expand));
        let bg_expand = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_v3_expand_bg"),
            layout: &l2_pipelines.expand_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: l2_bufs.group_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: l2_bufs.group_minx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: l2_bufs.group_miny.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: l2_bufs.group_maxx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: l2_bufs.group_maxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: l2_bufs.group_color.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: l2_bufs.params_expand.as_entire_binding() },
            ],
        });
        let pass_start = Instant::now();
        for _ in 0..cfg.l2_prop_iters.max(1) {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_v3_expand_enc") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_v3_expand_pass"), timestamp_writes: None });
                pass.set_pipeline(&l2_pipelines.expand_pipeline);
                pass.set_bind_group(0, &bg_expand, &[]);
                pass.dispatch_workgroups((l2_bufs.groups_count + 255) / 256, 1, 1);
            }
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::Wait);
        }
        l2_pass_times.push(("l2_expand", pass_start.elapsed()));

        let pass_start = Instant::now();
        let params_emit = [l2_bufs.groups_count, 0u32, 0u32, 0u32];
        queue.write_buffer(&l2_bufs.params_emit, 0, bytemuck::cast_slice(&params_emit));
        let bg_emit = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("l2_v3_emit_bg"),
            layout: &l2_pipelines.emit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: l2_bufs.group_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: l2_bufs.group_minx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: l2_bufs.group_miny.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: l2_bufs.group_maxx.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: l2_bufs.group_maxy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: l2_bufs.group_color.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: l2_bufs.out_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: l2_bufs.out_boxes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: l2_bufs.params_emit.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_v3_emit_enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_v3_emit_pass"), timestamp_writes: None });
            pass.set_pipeline(&l2_pipelines.emit_pipeline);
            pass.set_bind_group(0, &bg_emit, &[]);
            pass.dispatch_workgroups((l2_bufs.groups_count + 255) / 256, 1, 1);
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::Wait);
        l2_pass_times.push(("l2_emit", pass_start.elapsed()));
    }
    let l2_dur = l2_start.elapsed();

    if cfg.save_l1_segments {
        let segments = readback_segments(device, queue, &bufs.segments, total_segments);
        let color565_map = readback_color565_map(device, queue, &l0_out.color565, width * height);
        let _ = visualization::save_l1_segments_overlay(
            src_path,
            &segments,
            &color565_map,
            width as usize,
            height as usize,
            "l1",
        );
    }

    let l2_boxes = if let Some(l2_bufs) = l2_buffers.as_ref() {
        read_l2_boxes(device, queue, &l2_bufs.out_count, &l2_bufs.out_boxes, total_segments)
    } else {
        Vec::new()
    };
    if cfg.save_l2_boxes {
        let _ = visualization::save_l2_boxes_on_src(
            src_path,
            &l2_boxes,
            width as usize,
            height as usize,
            "l2",
        );
    }

    let l3_cfg = layer3_cpu::L3Config::from_app_config(cfg);
    let l3_boxes = layer3_cpu::l3_merge_and_refine_cpu(&l2_boxes, &l3_cfg);
    if cfg.save_l3_boxes {
        let _ = visualization::save_l3_boxes_on_src(
            src_path,
            &l3_boxes,
            width as usize,
            height as usize,
            "l3",
        );
    }

    if config::get().log_timing {
        visualization::log_timing_layers_l2_passes(
            src_path,
            info,
            l0_dur,
            l1_dur,
            l2_dur,
            total_start.elapsed(),
            &l2_pass_times,
        );
    }
}
