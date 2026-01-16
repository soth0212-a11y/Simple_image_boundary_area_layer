use std::path::Path;
use std::time::{Duration, Instant};
use wgpu;
use wgpu::util::DeviceExt;

use crate::config;
use crate::preprocessing;
use crate::visualization;
use crate::l1_bbox_gpu;

pub struct Layer0Outputs {
    pub packed: wgpu::Buffer,
    pub cell_rgb: wgpu::Buffer,
    pub edge4: wgpu::Buffer,
    pub s_active: wgpu::Buffer,
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
    (Layer0Outputs { packed: out_packed, cell_rgb, edge4, s_active }, info)
}

pub fn model(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img_view: wgpu::TextureView,
    img_info: preprocessing::Imginfo,
    l0_static: &layer_static_values,
    l1_pipelines: &l1_bbox_gpu::L1BboxPipelines,
    l1_buffers: &mut Option<l1_bbox_gpu::L1BboxBuffers>,
    src_path: &Path,
) {
    let total_start = Instant::now();

    let l0_start = Instant::now();
    let (l0_out, info) = layer0(device, queue, &img_view, img_info, l0_static);
    device.poll(wgpu::PollType::Wait);
    let l0_dur = l0_start.elapsed();

    let width = info[1];
    let height = info[0];
    let grid_w = width;
    let grid_h = height;
    if config::get().save_layer0 {
        let out_w = grid_w;
        let out_h = grid_h;
        let l0_bytes = (out_w * out_h) as u64 * 8;
        let packed = readback_u32_buffer(device, queue, &l0_out.packed, l0_bytes);
        let _ = visualization::save_layer0_packed_rgb(
            src_path,
            &packed,
            out_w as usize,
            out_h as usize,
            "l0",
        );
    }

    let cfg = config::get();
    let rebuild = l1_buffers
        .as_ref()
        .map(|b| b.w != grid_w || b.h != grid_h)
        .unwrap_or(true);
    if rebuild {
        *l1_buffers = Some(l1_bbox_gpu::ensure_l1_bbox_buffers(device, grid_w, grid_h));
    }
    let l1_bufs = l1_buffers.as_ref().unwrap();
    l1_bbox_gpu::dispatch_l1_bbox(
        device,
        queue,
        l1_pipelines,
        l1_bufs,
        &l0_out.s_active,
        &l0_out.packed,
        cfg.l1_enable_stride2,
    );
    device.poll(wgpu::PollType::Wait);

    if cfg.save_l1_bbox {
        let cell_bytes = (grid_w * grid_h) as u64 * 4;
        let bbox0 = readback_u32_buffer(device, queue, &l1_bufs.bbox0_s1, cell_bytes);
        let bbox1 = readback_u32_buffer(device, queue, &l1_bufs.bbox1_s1, cell_bytes);
        let colors = readback_u32_buffer(device, queue, &l1_bufs.color_s1, cell_bytes);
        let _ = visualization::save_l1_bbox_on_src(
            src_path,
            &bbox0,
            &bbox1,
            &colors,
            grid_w as usize,
            grid_h as usize,
            "s1",
        );
    }
    if cfg.l1_enable_stride2 && cfg.save_l1_bbox_stride2 {
        let s2_bytes = (l1_bufs.stride2_w * l1_bufs.stride2_h) as u64 * 4;
        let bbox0 = readback_u32_buffer(device, queue, &l1_bufs.bbox0_s2, s2_bytes);
        let bbox1 = readback_u32_buffer(device, queue, &l1_bufs.bbox1_s2, s2_bytes);
        let colors = readback_u32_buffer(device, queue, &l1_bufs.color_s2, s2_bytes);
        let _ = visualization::save_l1_bbox_on_src(
            src_path,
            &bbox0,
            &bbox1,
            &colors,
            l1_bufs.stride2_w as usize,
            l1_bufs.stride2_h as usize,
            "s2",
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
