use std::path::Path;
use std::time::{Duration, Instant};
use wgpu;
use wgpu::util::DeviceExt;

use crate::config;
use crate::preprocessing;
use crate::visualization; // 시각화 모듈 사용
use crate::l3_gpu;
use crate::l4_gpu;
use crate::l5;

// ---- 상수 및 구조체 정의 ----
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeDirCounts {
    pub top_left: i32, pub top_center: i32, pub top_right: i32,
    pub center_left: i32, pub center_right: i32,
    pub bottom_left: i32, pub bottom_center: i32, pub bottom_right: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedDirs { pub a: u32, pub b: u32 }

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Layer1Out { pub dir_flags: u32 }

pub struct Layer1Outputs {
    pub mask: wgpu::Buffer,
    pub out_r: wgpu::Buffer,
    pub out_g: wgpu::Buffer,
    pub out_b: wgpu::Buffer,
}

pub struct Layer2Outputs {
    pub mask: wgpu::Buffer,
    pub out_w: u32,
    pub out_h: u32,
}

pub struct Layer0Outputs {
    pub r: wgpu::Buffer,
    pub g: wgpu::Buffer,
    pub b: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec4i32 { pub x0: i32, pub y0: i32, pub x1: i32, pub y1: i32 }



pub struct layer_static_values {
    pub bindgroup_layout: wgpu::BindGroupLayout,
    pub compute_pipeline: wgpu::ComputePipeline,
}

pub struct layer2_static_values {
    pub bindgroup_layout: wgpu::BindGroupLayout,
    pub compute_pipeline: wgpu::ComputePipeline,
}

// ---- 유틸리티 함수 ----
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
    drop(data); staging.unmap();
    result
}


// ---- 파이프라인 초기화 (모든 레이어) ----

pub fn layer0_init(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> layer_static_values {
    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l0_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ][..],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bindgroup_layout], push_constant_ranges: &[][..] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    layer_static_values { bindgroup_layout, compute_pipeline }
}

pub fn layer1_init(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> layer_static_values {
    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l1_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ][..],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bindgroup_layout], push_constant_ranges: &[][..] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    layer_static_values { bindgroup_layout, compute_pipeline }
}

pub fn layer2_init(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> layer2_static_values {
    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l2_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ][..],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bindgroup_layout], push_constant_ranges: &[][..] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("l2_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    layer2_static_values { bindgroup_layout, compute_pipeline }
}

// ---- 레이어 실행 함수 ----

pub fn layer0(device: &wgpu::Device, queue: &wgpu::Queue, img_view: &wgpu::TextureView, img_info: preprocessing::Imginfo, static_vals: &layer_static_values) -> (Layer0Outputs, [u32; 4]) {
    let info = [img_info.height, img_info.width, img_info.height, img_info.width];
    let grid_w = img_info.width;
    let grid_h = img_info.height;
    let out_size = (grid_w * grid_h) as u64 * 4; // u32 mask per cell
    let out_r = device.create_buffer(&wgpu::BufferDescriptor { label: Some("layer0_out_r"), size: out_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let out_g = device.create_buffer(&wgpu::BufferDescriptor { label: Some("layer0_out_g"), size: out_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let out_b = device.create_buffer(&wgpu::BufferDescriptor { label: Some("layer0_out_b"), size: out_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let info_buf_data = [img_info.height, img_info.width];
    let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&info_buf_data), usage: wgpu::BufferUsages::STORAGE });
    
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &static_vals.bindgroup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(img_view) },
            wgpu::BindGroupEntry { binding: 1, resource: info_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_r.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_g.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out_b.as_entire_binding() },
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
    (Layer0Outputs { r: out_r, g: out_g, b: out_b }, info)
}


pub fn layer1(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    l0_out: &Layer0Outputs,
    img_info: [u32; 4],
    static_vals: &layer_static_values,
    src_path: &Path,
) -> Layer1Outputs {
    let height = img_info[0];
    let width = img_info[1];
    let out_w = (width + 1) / 2;
    let out_h = (height + 1) / 2;
    let dispatch_w = out_w;
    let dispatch_h = out_h;
    let info = [height, width];
    let n_cells = dispatch_w * dispatch_h;
    let buf_size = (n_cells * 4) as u64;
    let out_mask = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer1_out_mask"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_r = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer1_out_r"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_g = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer1_out_g"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer1_out_b"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let zero = vec![0u8; buf_size as usize];
    queue.write_buffer(&out_mask, 0, &zero);
    queue.write_buffer(&out_r, 0, &zero);
    queue.write_buffer(&out_g, 0, &zero);
    queue.write_buffer(&out_b, 0, &zero);

    let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("layer1_info"),
        contents: bytemuck::cast_slice(&info),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &static_vals.bindgroup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: info_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: l0_out.r.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: l0_out.g.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: l0_out.b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: out_r.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: out_g.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: out_b.as_entire_binding() },
        ][..],
        label: Some("layer1_bg"),
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("layer1_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("layer1_pass"), timestamp_writes: None });
        pass.set_pipeline(&static_vals.compute_pipeline);
        pass.set_bind_group(0, &bg, &[][..][..]);
        pass.dispatch_workgroups((dispatch_w + 15) / 16, (dispatch_h + 15) / 16, 1);
    }
    queue.submit([enc.finish()]);

    if config::get().save_layer1 {
        let mask = readback_u32_buffer(device, queue, &out_mask, buf_size);
        let _ = visualization::save_layer1_mask_overlay(
            src_path,
            &mask,
            out_w as usize,
            out_h as usize,
            "mask",
        );
        let r = readback_u32_buffer(device, queue, &out_r, buf_size);
        let g = readback_u32_buffer(device, queue, &out_g, buf_size);
        let b = readback_u32_buffer(device, queue, &out_b, buf_size);
        let _ = visualization::save_layer1_channel(src_path, &mask, &r, out_w as usize, out_h as usize, "val", "r");
        let _ = visualization::save_layer1_channel(src_path, &mask, &g, out_w as usize, out_h as usize, "val", "g");
        let _ = visualization::save_layer1_channel(src_path, &mask, &b, out_w as usize, out_h as usize, "val", "b");
        let _ = visualization::save_layer1_rgb_composite(
            src_path,
            &mask,
            &r,
            &g,
            &b,
            out_w as usize,
            out_h as usize,
            "val",
        );
    }
    Layer1Outputs {
        mask: out_mask,
        out_r,
        out_g,
        out_b,
    }
}

pub fn layer2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    l1: &Layer1Outputs,
    img_info: [u32; 4],
    static_vals: &layer2_static_values,
) -> Layer2Outputs {
    const KERNEL: u32 = 2;
    const STRIDE: u32 = 1;

    let height = img_info[0];
    let width = img_info[1];
    if width == 0 || height == 0 {
        return Layer2Outputs {
            mask: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("l2_pool_mask_empty"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            out_w: 0,
            out_h: 0,
        };
    }

    let l1_w = (width + 1) / 2;
    let l1_h = (height + 1) / 2;
    let (out_w, out_h) = if l1_w < 2 || l1_h < 2 {
        (1u32, 1u32)
    } else {
        let ow = ((l1_w + STRIDE - 1 - KERNEL) / STRIDE) + 1;
        let oh = ((l1_h + STRIDE - 1 - KERNEL) / STRIDE) + 1;
        (ow.max(1), oh.max(1))
    };

    let info = [height, width];
    let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("l2_info"),
        contents: bytemuck::cast_slice(&info),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_len = (out_w * out_h) as u64;
    let out_size = out_len * 4;
    let pooled_mask = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_pool_mask"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let zero = vec![0u8; out_size as usize];
    queue.write_buffer(&pooled_mask, 0, &zero);

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &static_vals.bindgroup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: info_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: l1.mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pooled_mask.as_entire_binding() },
        ][..],
        label: Some("l2_pool_bg"),
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l2_pool_enc") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l2_pool_pass"), timestamp_writes: None });
        pass.set_pipeline(&static_vals.compute_pipeline);
        pass.set_bind_group(0, &bg, &[][..]);
        pass.dispatch_workgroups((out_w + 15) / 16, (out_h + 15) / 16, 1);
    }
    queue.submit([enc.finish()]);

    Layer2Outputs {
        mask: pooled_mask,
        out_w,
        out_h,
    }
}

// ---- 메인 모델 조정 함수 ----

pub fn model(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img_view: wgpu::TextureView,
    img_info: preprocessing::Imginfo,
    static_vals: [&layer_static_values; 2], // [l0..l1]
    l2_static: &layer2_static_values,
    l3_pipelines: &l3_gpu::L3Pipelines,
    l3_buffers: &mut Option<l3_gpu::L3Buffers>,
    l4_pipelines: &l4_gpu::L4ChannelsPipelines,
    l4_buffers: &mut Option<l4_gpu::L4ChannelsBuffers>,
    l5_pipelines: &l5::L5Pipelines,
    l5_buffers: &mut Option<l5::L5Buffers>,
    src_path: &Path,
) {
    let total_start = Instant::now();
    let mut l3_dur = Duration::from_secs(0);
    let mut l4_dur = Duration::from_secs(0);
    let mut l5_dur = Duration::from_secs(0);

    // ---- Layer0 ----
    let l0_start = Instant::now();
    let (l0_out, info) = layer0(device, queue, &img_view, img_info, static_vals[0]);
    device.poll(wgpu::PollType::Wait);
    let l0_dur = l0_start.elapsed();
    // ---- Layer1 ----
    let l1_start = Instant::now();
    let l1_out = layer1(device, queue, &l0_out, info, static_vals[1], src_path);
    device.poll(wgpu::PollType::Wait);
    let l1_dur = l1_start.elapsed();

    let l2_start = Instant::now();
    let l2_out = layer2(device, queue, &l1_out, info, l2_static);
    device.poll(wgpu::PollType::Wait);
    let l2_dur = l2_start.elapsed();

    let height = info[0];
    let width = info[1];
    if config::get().save_layer2 && l2_out.out_w > 0 && l2_out.out_h > 0 {
        let out_size = (l2_out.out_w * l2_out.out_h) as u64 * 4;
        let pooled = readback_u32_buffer(device, queue, &l2_out.mask, out_size);
        let _ = visualization::save_layer2_overlay(
            src_path,
            &pooled,
            l2_out.out_w as usize,
            l2_out.out_h as usize,
            width as usize,
            height as usize,
            "layer2",
        );
    }

    if l2_out.out_w > 0 && l2_out.out_h > 0 {
        let l3_in_w = l2_out.out_w;
        let l3_in_h = l2_out.out_h;
        let l3_out_w = (l3_in_w + 1) / 2;
        let l3_out_h = (l3_in_h + 1) / 2;
        let rebuild = l3_buffers
            .as_ref()
            .map(|b| b.in_w != l3_in_w || b.in_h != l3_in_h || b.out_w != l3_out_w || b.out_h != l3_out_h)
            .unwrap_or(true);
        if rebuild {
            *l3_buffers = Some(l3_gpu::ensure_l3_buffers(device, l3_in_w, l3_in_h, l3_out_w, l3_out_h));
        }
        let l3_bufs = l3_buffers.as_ref().unwrap();
        let l3_start = Instant::now();
        l3_gpu::dispatch_l3(device, queue, l3_pipelines, l3_bufs, &l2_out.mask);
        device.poll(wgpu::PollType::Wait);
        l3_dur = l3_start.elapsed();

        if l3_out_w > 0 && l3_out_h > 0 {
            let l3_size = (l3_out_w * l3_out_h) as u64 * 4;
            let need_l3 = config::get().save_layer3 || config::get().log_layer3;
            if need_l3 {
                let s_active = readback_u32_buffer(device, queue, &l3_bufs.s_active, l3_size);
                let conn8 = readback_u32_buffer(device, queue, &l3_bufs.conn8, l3_size);
                if config::get().save_layer3 {
                    let _ = visualization::save_layer3_s_active_overlay(
                        src_path,
                        &s_active,
                        l3_out_w as usize,
                        l3_out_h as usize,
                        "l3",
                    );
                    let _ = visualization::save_layer3_conn8_overlay(
                        src_path,
                        &conn8,
                        l3_out_w as usize,
                        l3_out_h as usize,
                        "l3",
                    );
                }
                if config::get().log_layer3 {
                    let _ = visualization::log_layer3_conn8(
                        src_path,
                        &conn8,
                        l3_out_w as usize,
                        l3_out_h as usize,
                    );
                }
            }

            let rebuild = l4_buffers
                .as_ref()
                .map(|b| b.w != l3_out_w || b.h != l3_out_h)
                .unwrap_or(true);
            if rebuild {
                *l4_buffers = Some(l4_gpu::ensure_l4_channels_buffers(device, l3_out_w, l3_out_h));
            }
            let l4_bufs = l4_buffers.as_ref().unwrap();
            let l4_start = Instant::now();
            l4_gpu::dispatch_l4_channels(device, queue, l4_pipelines, l4_bufs, &l3_bufs.s_active, &l3_bufs.conn8);
            device.poll(wgpu::PollType::Wait);
            l4_dur = l4_start.elapsed();

            if config::get().save_layer4 {
                let bbox0 = readback_u32_buffer(device, queue, &l4_bufs.bbox0, l3_size);
                let bbox1 = readback_u32_buffer(device, queue, &l4_bufs.bbox1, l3_size);
                let meta = readback_u32_buffer(device, queue, &l4_bufs.meta, l3_size);
                let _ = visualization::save_layer4_expand_mask(
                    src_path,
                    &meta,
                    l3_out_w as usize,
                    l3_out_h as usize,
                    "l4",
                );
                let _ = visualization::save_layer4_bbox_overlay(
                    src_path,
                    &bbox0,
                    &bbox1,
                    &meta,
                    l3_out_w as usize,
                    l3_out_h as usize,
                    "l4",
                );
                let _ = visualization::log_layer4_packed(
                    src_path,
                    &bbox0,
                    &bbox1,
                    &meta,
                    l3_out_w as usize,
                    l3_out_h as usize,
                );
            }

            let l5_rebuild = l5_buffers
                .as_ref()
                .map(|b| b.w != l3_out_w || b.h != l3_out_h)
                .unwrap_or(true);
            if l5_rebuild {
                *l5_buffers = Some(l5::ensure_l5_buffers(device, l3_out_w, l3_out_h));
            }
            let l5_bufs = l5_buffers.as_ref().unwrap();
            let threshold = l5::default_threshold();
            let iterations = l5::default_iters(l3_out_w, l3_out_h);
            let l5_start = Instant::now();
            let final_is_a = l5::dispatch_l5(
                device,
                queue,
                l5_pipelines,
                l5_bufs,
                &l3_bufs.s_active,
                &l3_bufs.conn8,
                threshold,
                iterations,
            );
            device.poll(wgpu::PollType::Wait);
            l5_dur = l5_start.elapsed();

            if config::get().save_l5_debug {
                let tile_bytes = (l5_bufs.tile_w * l5_bufs.tile_h) as u64 * 4;
                let cell_bytes = (l5_bufs.w * l5_bufs.h) as u64 * 4;
                let score = readback_u32_buffer(device, queue, &l5_bufs.score_map, tile_bytes);
                let keep = readback_u32_buffer(device, queue, &l5_bufs.tile_keep, tile_bytes);
                let roi = readback_u32_buffer(device, queue, &l5_bufs.roi_mask, cell_bytes);
                let label_buf = if final_is_a { &l5_bufs.label_a } else { &l5_bufs.label_b };
                let labels = readback_u32_buffer(device, queue, label_buf, cell_bytes);
                let _ = visualization::save_l5_score_map(
                    src_path,
                    &score,
                    l5_bufs.tile_w as usize,
                    l5_bufs.tile_h as usize,
                    "l5",
                );
                let _ = visualization::save_l5_tile_keep(
                    src_path,
                    &keep,
                    l5_bufs.tile_w as usize,
                    l5_bufs.tile_h as usize,
                    "l5",
                );
                let _ = visualization::save_l5_roi_mask(
                    src_path,
                    &roi,
                    l5_bufs.w as usize,
                    l5_bufs.h as usize,
                    "l5",
                );
                let _ = visualization::save_l5_label_map(
                    src_path,
                    &labels,
                    l5_bufs.w as usize,
                    l5_bufs.h as usize,
                    "l5",
                );
            }
        }
    }

    let grid3_w = width;
    let grid3_h = height;
    if config::get().save_layer0 {
        let l0_bytes = (grid3_w * grid3_h) as u64 * 4;
        let raw_r = readback_u32_buffer(device, queue, &l0_out.r, l0_bytes);
        let raw_g = readback_u32_buffer(device, queue, &l0_out.g, l0_bytes);
        let raw_b = readback_u32_buffer(device, queue, &l0_out.b, l0_bytes);
        let _ = visualization::save_layer0_channel_overlay(
            src_path,
            &raw_r,
            grid3_w as usize,
            grid3_h as usize,
            "mask",
            "r",
        );
        let _ = visualization::save_layer0_channel_overlay(
            src_path,
            &raw_g,
            grid3_w as usize,
            grid3_h as usize,
            "mask",
            "g",
        );
        let _ = visualization::save_layer0_channel_overlay(
            src_path,
            &raw_b,
            grid3_w as usize,
            grid3_h as usize,
            "mask",
            "b",
        );
    }

    if config::get().log_timing {
        visualization::log_timing_layers(
            src_path,
            info,
            l0_dur,
            l1_dur,
            l2_dur,
            l3_dur,
            l4_dur,
            l5_dur,
            total_start.elapsed(),
        );
    }
}
