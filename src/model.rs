use std::path::Path;
use std::time::Instant;
use wgpu;
use wgpu::util::DeviceExt;

use crate::config;
use crate::preprocessing;
use crate::visualization; // 시각화 모듈 사용
use crate::l3_gpu;
use crate::l4_gpu;

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
}

pub struct Layer2Outputs {
    pub mask: wgpu::Buffer,
    pub out_w: u32,
    pub out_h: u32,
}

#[derive(Clone, Copy, Default)]
pub struct BBox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32, // exclusive
}

#[derive(Clone, Copy, Default)]
pub struct BBoxScore {
    pub bbox: BBox,
    pub score: u64,
    pub area: u64,
    pub flags: u32,
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

pub struct L4Pipelines {
    pub flatten: layer_static_values,
    pub bin_fill: layer_static_values,
    pub merge_offset: layer_static_values,
    pub reduce: layer_static_values,
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
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
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

pub fn layer4_flatten_init(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> layer_static_values {
    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l4_flatten_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ][..],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bindgroup_layout], push_constant_ranges: &[][..] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("l4_flatten_pipeline"), layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    layer_static_values { bindgroup_layout, compute_pipeline }
}

pub fn layer4_bin_init(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> layer_static_values {
    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l4_bin_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ][..],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bindgroup_layout], push_constant_ranges: &[][..] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("l4_bin_pipeline"), layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    layer_static_values { bindgroup_layout, compute_pipeline }
}

pub fn layer4_merge_init(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> layer_static_values {
    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l4_merge_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ][..],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bindgroup_layout], push_constant_ranges: &[][..] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("l4_merge_pipeline"), layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    layer_static_values { bindgroup_layout, compute_pipeline }
}

pub fn layer4_reduce_init(device: &wgpu::Device, shader_module: wgpu::ShaderModule) -> layer_static_values {
    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l4_reduce_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ][..],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bindgroup_layout], push_constant_ranges: &[][..] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("l4_reduce_pipeline"), layout: Some(&pipeline_layout), module: &shader_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    layer_static_values { bindgroup_layout, compute_pipeline }
}


// ---- 레이어 실행 함수 ----

pub fn layer0(device: &wgpu::Device, queue: &wgpu::Queue, img_view: wgpu::TextureView, img_info: preprocessing::Imginfo, static_vals: &layer_static_values) -> (wgpu::Buffer, [u32; 4]) {
    let info = [img_info.height, img_info.width, img_info.height, img_info.width];
    let grid_w = img_info.width;
    let grid_h = img_info.height;
    let out_size = (grid_w * grid_h) as u64 * 4; // u32 mask per cell
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: out_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let info_buf_data = [img_info.height, img_info.width];
    let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&info_buf_data), usage: wgpu::BufferUsages::STORAGE });
    
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &static_vals.bindgroup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&img_view) },
            wgpu::BindGroupEntry { binding: 1, resource: info_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
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
    (out_buf, info)
}

pub fn layer1(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    l0_buf: &wgpu::Buffer,
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
    let zero = vec![0u8; buf_size as usize];
    queue.write_buffer(&out_mask, 0, &zero);

    let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("layer1_info"),
        contents: bytemuck::cast_slice(&info),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &static_vals.bindgroup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: info_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: l0_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_mask.as_entire_binding() },
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
    }
    Layer1Outputs {
        mask: out_mask,
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

struct DSU {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DSU {
    fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        for i in 0..n {
            parent.push(i);
        }
        Self { parent, rank: vec![0; n] }
    }

    fn find(&mut self, x: usize) -> usize {
        let p = self.parent[x];
        if p != x {
            let root = self.find(p);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let ra_rank = self.rank[ra];
        let rb_rank = self.rank[rb];
        if ra_rank < rb_rank {
            self.parent[ra] = rb;
        } else if ra_rank > rb_rank {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] = ra_rank + 1;
        }
    }
}

const FLAG_TOUCH_N: u32 = 16u32;
const FLAG_TOUCH_E: u32 = 32u32;
const FLAG_TOUCH_S: u32 = 64u32;
const FLAG_TOUCH_W: u32 = 128u32;

fn iou_ge_10(a: BBox, b: BBox) -> bool {
    if a.x1 <= a.x0 || a.y1 <= a.y0 || b.x1 <= b.x0 || b.y1 <= b.y0 {
        return false;
    }
    let inter_x0 = a.x0.max(b.x0);
    let inter_y0 = a.y0.max(b.y0);
    let inter_x1 = a.x1.min(b.x1);
    let inter_y1 = a.y1.min(b.y1);
    if inter_x1 <= inter_x0 || inter_y1 <= inter_y0 {
        return false;
    }
    let inter_w = (inter_x1 - inter_x0) as u64;
    let inter_h = (inter_y1 - inter_y0) as u64;
    let inter_area = inter_w * inter_h;
    let area_a = ((a.x1 - a.x0) as u64) * ((a.y1 - a.y0) as u64);
    let area_b = ((b.x1 - b.x0) as u64) * ((b.y1 - b.y0) as u64);
    let union_area = area_a + area_b - inter_area;
    if union_area == 0 {
        return false;
    }
    inter_area * 10 >= union_area
}

fn l4_allow_merge(a: &BBoxScore, b: &BBoxScore) -> bool {
    let ax = (a.bbox.x0 + a.bbox.x1) / 2;
    let ay = (a.bbox.y0 + a.bbox.y1) / 2;
    let bx = (b.bbox.x0 + b.bbox.x1) / 2;
    let by = (b.bbox.y0 + b.bbox.y1) / 2;
    let dx = bx as i64 - ax as i64;
    let dy = by as i64 - ay as i64;
    if dx == 0 && dy == 0 {
        return true;
    }

    let a_n = (a.flags & FLAG_TOUCH_N) != 0u32;
    let a_e = (a.flags & FLAG_TOUCH_E) != 0u32;
    let a_s = (a.flags & FLAG_TOUCH_S) != 0u32;
    let a_w = (a.flags & FLAG_TOUCH_W) != 0u32;
    let b_n = (b.flags & FLAG_TOUCH_N) != 0u32;
    let b_e = (b.flags & FLAG_TOUCH_E) != 0u32;
    let b_s = (b.flags & FLAG_TOUCH_S) != 0u32;
    let b_w = (b.flags & FLAG_TOUCH_W) != 0u32;

    if dx == 0 {
        if dy > 0 { return a_s && b_n; }
        return a_n && b_s;
    }
    if dy == 0 {
        if dx > 0 { return a_e && b_w; }
        return a_w && b_e;
    }

    if dx > 0 && dy < 0 { return a_n && a_e && b_s && b_w; }
    if dx > 0 && dy > 0 { return a_s && a_e && b_n && b_w; }
    if dx < 0 && dy < 0 { return a_n && a_w && b_s && b_e; }
    a_s && a_w && b_n && b_e
}

fn l4_union_merge_once(boxes: &[BBoxScore]) -> (Vec<BBoxScore>, bool) {
    let n = boxes.len();
    if n == 0 {
        return (Vec::new(), false);
    }
    let mut dsu = DSU::new(n);
    let radius_sq: u64 = 8u64 * 8u64;
    let mut changed = false;
    for i in 0..n {
        for j in (i + 1)..n {
            if center_distance_sq(boxes[i].bbox, boxes[j].bbox) <= radius_sq
                && iou_ge_10(boxes[i].bbox, boxes[j].bbox)
                && l4_allow_merge(&boxes[i], &boxes[j])
            {
                let ra = dsu.find(i);
                let rb = dsu.find(j);
                if ra != rb {
                    dsu.union(ra, rb);
                    changed = true;
                }
            }
        }
    }
    let mut agg: std::collections::HashMap<usize, BBoxScore> = std::collections::HashMap::new();
    for i in 0..n {
        let root = dsu.find(i);
        let b = boxes[i];
        let entry = agg.entry(root).or_insert(BBoxScore {
            bbox: b.bbox,
            score: 0,
            area: 0,
            flags: 0,
        });
        entry.bbox.x0 = entry.bbox.x0.min(b.bbox.x0);
        entry.bbox.y0 = entry.bbox.y0.min(b.bbox.y0);
        entry.bbox.x1 = entry.bbox.x1.max(b.bbox.x1);
        entry.bbox.y1 = entry.bbox.y1.max(b.bbox.y1);
        entry.score = entry.score.saturating_add(b.score);
        entry.flags |= b.flags;
    }
    let mut out: Vec<BBoxScore> = Vec::new();
    for (_root, mut v) in agg {
        let w = (v.bbox.x1.saturating_sub(v.bbox.x0)) as u64;
        let h = (v.bbox.y1.saturating_sub(v.bbox.y0)) as u64;
        v.area = w * h;
        out.push(v);
    }
    (out, changed)
}

fn l4_union_merge(boxes: &[BBoxScore]) -> Vec<BBoxScore> {
    let mut current: Vec<BBoxScore> = boxes.to_vec();
    loop {
        let (next, changed) = l4_union_merge_once(&current);
        if !changed || next.len() == current.len() {
            return next;
        }
        current = next;
    }
}

fn center_distance_sq(a: BBox, b: BBox) -> u64 {
    let ax = (a.x0 + a.x1) / 2;
    let ay = (a.y0 + a.y1) / 2;
    let bx = (b.x0 + b.x1) / 2;
    let by = (b.y0 + b.y1) / 2;
    let dx = ax as i64 - bx as i64;
    let dy = ay as i64 - by as i64;
    (dx * dx + dy * dy) as u64
}

// ---- 메인 모델 조정 함수 ----

pub fn model(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img_view: wgpu::TextureView,
    img_info: preprocessing::Imginfo,
    static_vals: [&layer_static_values; 2], // [l0..l1]
    l2_static: &layer2_static_values,
    l3_pipelines: &l3_gpu::L3GpuPipelines,
    l3_buffers: &mut Option<l3_gpu::L3GpuBuffers>,
    l4_pipelines: &L4Pipelines,
    l4_buffers: &mut Option<l4_gpu::L4GpuBuffers>,
    src_path: &Path,
) {
    let total_start = Instant::now();
    let mut l3_e = std::time::Duration::from_secs(0);
    let mut l4_e = std::time::Duration::from_secs(0);

    // ---- Layer0 ----
    let l0_s = Instant::now();
    let (l0_dirs, info) = layer0(device, queue, img_view, img_info, static_vals[0]);
    let l0_e = l0_s.elapsed();

    // ---- Layer1 ----
    let l1_s = Instant::now();
    let l1_out = layer1(device, queue, &l0_dirs, info, static_vals[1], src_path);
    let l1_e = l1_s.elapsed();

    let l2_s = Instant::now();
    let l2_out = layer2(device, queue, &l1_out, info, l2_static);
    let l2_e = l2_s.elapsed();

    let height = info[0];
    let width = info[1];
    let mut pooled_cpu: Option<Vec<u32>> = None;
    if (config::get().save_layer2 || config::get().save_layer3 || config::get().save_layer4) && l2_out.out_w > 0 && l2_out.out_h > 0 {
        let out_size = (l2_out.out_w * l2_out.out_h) as u64 * 4;
        let pooled = readback_u32_buffer(device, queue, &l2_out.mask, out_size);
        if config::get().save_layer2 {
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
        pooled_cpu = Some(pooled);
    }

    let need_l3 = config::get().save_layer3 || config::get().save_layer4 || config::get().log_timing;
    if need_l3 && l2_out.out_w >= 16 && l2_out.out_h >= 16 {
        let l3_s = Instant::now();
        let aw = ((l2_out.out_w - 16) / 4) + 1;
        let ah = ((l2_out.out_h - 16) / 4) + 1;
        let bw = (aw + 3) / 4;
        let bh = (ah + 3) / 4;
        let bw2 = (bw + 1) / 2;
        let bh2 = (bh + 1) / 2;
        if aw > 0 && ah > 0 && bw > 0 && bh > 0 && bw2 > 0 && bh2 > 0 {
            let rebuild = l3_buffers
                .as_ref()
                .map(|b| b.aw != aw || b.ah != ah || b.bw != bw || b.bh != bh || b.bw2 != bw2 || b.bh2 != bh2)
                .unwrap_or(true);
            if rebuild {
                *l3_buffers = Some(l3_gpu::ensure_l3_buffers(device, aw, ah, bw, bh, bw2, bh2));
            }
            let l3_bufs = l3_buffers.as_ref().unwrap();

            let info_buf_data = [height, width];
            let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("l3_info"),
                contents: bytemuck::cast_slice(&info_buf_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l3_enc") });
            l3_gpu::dispatch_l3(
                device,
                &mut enc,
                l3_pipelines,
                l3_bufs,
                &info_buf,
                &l2_out.mask,
                aw,
                ah,
                bw,
                bh,
                bw2,
                bh2,
            );
            queue.submit([enc.finish()]);

            let block_count = (bw2 * bh2) as usize;
            let block_slots = block_count * 16;
            let mut block_boxes: Vec<u32> = Vec::new();
            let mut block_valid: Vec<u32> = Vec::new();

            if config::get().save_layer3 && block_slots > 0 {
                let boxes_bytes = (block_slots * 2 * 4) as u64;
                block_boxes = readback_u32_buffer(device, queue, &l3_bufs.block_boxes, boxes_bytes);
                let valid_bytes = (block_slots * 4) as u64;
                block_valid = readback_u32_buffer(device, queue, &l3_bufs.block_valid, valid_bytes);
            }

            if config::get().save_layer3 {
                let anchor_count = (aw * ah) as usize;
                if anchor_count > 0 {
                    let anchor_bbox_bytes = (anchor_count * 2 * 4) as u64;
                    let anchor_score_bytes = (anchor_count * 4) as u64;
                    let anchor_bbox = readback_u32_buffer(device, queue, &l3_bufs.anchor_bbox, anchor_bbox_bytes);
                    let anchor_score = readback_u32_buffer(device, queue, &l3_bufs.anchor_score, anchor_score_bytes);
                    let anchor_meta = readback_u32_buffer(device, queue, &l3_bufs.anchor_meta, anchor_score_bytes);

                    let _ = visualization::save_layer3_pass1_anchor_overlay(
                        src_path,
                        l2_out.out_w as usize,
                        l2_out.out_h as usize,
                        aw as usize,
                        ah as usize,
                        &anchor_bbox,
                        &anchor_score,
                        &anchor_meta,
                    );
                }

                if block_slots > 0 {
                    if let Some(ref pooled) = pooled_cpu {
                        let _ = visualization::save_layer3_block_overlay(
                            src_path,
                            pooled,
                            l2_out.out_w as usize,
                            l2_out.out_h as usize,
                            &block_boxes,
                            &block_valid,
                            "l3",
                        );
                    }
                }
            }

            l3_e = l3_s.elapsed();

            if block_slots > 0 {
                let l4_s = Instant::now();

                let nmax = block_slots as u32;
                let grid_w = (l2_out.out_w + 40) / 41;
                let grid_h = (l2_out.out_h + 40) / 41;
                let rebuild = l4_buffers
                    .as_ref()
                    .map(|b| b.nmax != nmax || b.grid_w != grid_w || b.grid_h != grid_h)
                    .unwrap_or(true);
                if rebuild {
                    *l4_buffers = Some(l4_gpu::ensure_l4_buffers(device, nmax, grid_w, grid_h));
                }
                let l4_bufs = l4_buffers.as_ref().unwrap();

                queue.write_buffer(&l4_bufs.flat_count, 0, bytemuck::cast_slice(&[0u32]));

                let flatten_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &l4_pipelines.flatten.bindgroup_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: l3_bufs.block_boxes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: l3_bufs.block_valid.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: l4_bufs.flat_boxes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: l4_bufs.flat_count.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: l4_bufs.label.as_entire_binding() },
                    ],
                    label: Some("l4_flatten_bg"),
                });

                let dispatch_x = (nmax + 255) / 256;
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l4_flatten_enc") });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l4_flatten_pass"), timestamp_writes: None });
                    pass.set_pipeline(&l4_pipelines.flatten.compute_pipeline);
                    pass.set_bind_group(0, &flatten_bg, &[][..]);
                    pass.dispatch_workgroups(dispatch_x, 1, 1);
                }
                queue.submit([enc.finish()]);

                let bin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &l4_pipelines.bin_fill.bindgroup_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: info_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: l4_bufs.flat_boxes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: l4_bufs.flat_count.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: l4_bufs.params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: l4_bufs.cell_counts.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: l4_bufs.cell_items.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: l4_bufs.overflow_flags.as_entire_binding() },
                    ],
                    label: Some("l4_bin_bg"),
                });

                let merge_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &l4_pipelines.merge_offset.bindgroup_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: info_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: l4_bufs.flat_boxes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: l4_bufs.flat_count.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: l4_bufs.params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: l4_bufs.cell_counts.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: l4_bufs.cell_items.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: l4_bufs.label.as_entire_binding() },
                    ],
                    label: Some("l4_merge_bg"),
                });

                let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &l4_pipelines.reduce.bindgroup_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: l4_bufs.flat_boxes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: l4_bufs.flat_count.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: l4_bufs.label.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: l4_bufs.out_boxes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: l4_bufs.out_valid.as_entire_binding() },
                    ],
                    label: Some("l4_reduce_bg"),
                });

                let zero_cells = vec![0u8; (l4_bufs.cell_count as usize) * 4];
                let offsets: [(i32, i32); 4] = [(0, 0), (0, 3), (3, 0), (3, 3)];
                for (ox, oy) in offsets {
                    let params = [ox, oy];
                    queue.write_buffer(&l4_bufs.params, 0, bytemuck::cast_slice(&params));
                    queue.write_buffer(&l4_bufs.cell_counts, 0, &zero_cells);
                    queue.write_buffer(&l4_bufs.overflow_flags, 0, &zero_cells);

                    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l4_bin_merge_enc") });
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l4_bin_pass"), timestamp_writes: None });
                        pass.set_pipeline(&l4_pipelines.bin_fill.compute_pipeline);
                        pass.set_bind_group(0, &bin_bg, &[][..]);
                        pass.dispatch_workgroups(dispatch_x, 1, 1);
                    }
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l4_merge_pass"), timestamp_writes: None });
                        pass.set_pipeline(&l4_pipelines.merge_offset.compute_pipeline);
                        pass.set_bind_group(0, &merge_bg, &[][..]);
                        pass.dispatch_workgroups(dispatch_x, 1, 1);
                    }
                    queue.submit([enc.finish()]);

                    if config::get().save_layer4 {
                        let overflow_bytes = (l4_bufs.cell_count as u64) * 4;
                        let overflow = readback_u32_buffer(device, queue, &l4_bufs.overflow_flags, overflow_bytes);
                        let overflowed = overflow.iter().filter(|&&v| v != 0u32).count();
                        if overflowed > 0 {
                            eprintln!("l4 overflow cells (offset {}, {}): {}", ox, oy, overflowed);
                        }
                    }
                }

                let mut out_boxes_init = vec![0u32; nmax as usize * 2];
                for i in 0..nmax as usize {
                    out_boxes_init[i * 2] = 0xFFFF_FFFFu32;
                }
                let out_valid_zero = vec![0u8; nmax as usize * 4];
                queue.write_buffer(&l4_bufs.out_boxes, 0, bytemuck::cast_slice(&out_boxes_init));
                queue.write_buffer(&l4_bufs.out_valid, 0, &out_valid_zero);

                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l4_reduce_enc") });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l4_reduce_pass"), timestamp_writes: None });
                    pass.set_pipeline(&l4_pipelines.reduce.compute_pipeline);
                    pass.set_bind_group(0, &reduce_bg, &[][..]);
                    pass.dispatch_workgroups(dispatch_x, 1, 1);
                }
                queue.submit([enc.finish()]);

                l4_e = l4_s.elapsed();

                if config::get().save_layer4 && l2_out.out_w > 0 && l2_out.out_h > 0 {
                    if pooled_cpu.is_none() {
                        let out_size = (l2_out.out_w * l2_out.out_h) as u64 * 4;
                        pooled_cpu = Some(readback_u32_buffer(device, queue, &l2_out.mask, out_size));
                    }
                    if let Some(ref pooled) = pooled_cpu {
                        let out_boxes_bytes = (nmax as u64) * 2 * 4;
                        let out_valid_bytes = (nmax as u64) * 4;
                        let out_boxes = readback_u32_buffer(device, queue, &l4_bufs.out_boxes, out_boxes_bytes);
                        let out_valid = readback_u32_buffer(device, queue, &l4_bufs.out_valid, out_valid_bytes);
                        let mut l4_boxes: Vec<(u32, u32, u32, u32)> = Vec::new();
                        for i in 0..(nmax as usize) {
                            if out_valid.get(i).copied().unwrap_or(0) == 0 {
                                continue;
                            }
                            let bidx = i * 2;
                            if bidx + 1 >= out_boxes.len() {
                                break;
                            }
                            let b0 = out_boxes[bidx];
                            let b1 = out_boxes[bidx + 1];
                            let x0 = (b0 & 0xFFFF) as u32;
                            let y0 = (b0 >> 16) as u32;
                            let x1 = (b1 & 0xFFFF) as u32;
                            let y1 = (b1 >> 16) as u32;
                            if x1 <= x0 || y1 <= y0 {
                                continue;
                            }
                            l4_boxes.push((x0, y0, x1, y1));
                        }
                        let _ = visualization::save_layer4_boxes_overlay(
                            src_path,
                            pooled,
                            l2_out.out_w as usize,
                            l2_out.out_h as usize,
                            &l4_boxes,
                            "l4",
                        );
                    }
                }
            }
        }
    }
    let grid3_w = width;
    let grid3_h = height;
    if config::get().save_layer0 {
        let l0_bytes = (grid3_w * grid3_h) as u64 * 4;
        let raw = readback_u32_buffer(device, queue, &l0_dirs, l0_bytes);
        let _ = visualization::save_layer0_dir_overlay(
            src_path,
            &raw,
            grid3_w as usize,
            grid3_h as usize,
            "mask",
        );
    }

    // ---- Timing log ----
    // 기존 log_timing_block이 l5를 안 받으면, 새 함수 만들거나, l4까지만 찍고 l5는 별도 로그로 남겨도 됨.
    if config::get().log_timing {
        visualization::log_timing_block(
            src_path,
            info,
            l0_e,
            l1_e,
            l2_e,
            l3_e,
            l4_e,
            total_start.elapsed(),
        );
    }

    // 필요하면 추가로 레이어 시간을 별도 로그로 남기세요.
}
