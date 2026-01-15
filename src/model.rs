use std::path::Path;
use std::time::Instant;
use wgpu;
use wgpu::util::DeviceExt;

use crate::config;
use crate::preprocessing;
use crate::visualization; // 시각화 모듈 사용
use crate::l3_edge_gpu;
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
    pub inactive_avg: wgpu::Buffer,
}

pub struct Layer2Outputs {
    pub mask: wgpu::Buffer,
    pub inactive_avg: wgpu::Buffer,
    pub out_w: u32,
    pub out_h: u32,
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

fn decode_bbox_pair(pair0: u32, pair1: u32) -> Option<(u32, u32, u32, u32)> {
    let x0 = pair0 & 0xFFFFu32;
    let y0 = pair0 >> 16u32;
    let x1 = pair1 & 0xFFFFu32;
    let y1 = pair1 >> 16u32;
    if x1 < x0 || y1 < y0 {
        return None;
    }
    Some((x0, y0, x1, y1))
}

fn iou_ge_tenths(a: (u32, u32, u32, u32), b: (u32, u32, u32, u32), tenths: u32) -> bool {
    let (ax0, ay0, ax1, ay1) = a;
    let (bx0, by0, bx1, by1) = b;
    let inter_x0 = ax0.max(bx0);
    let inter_y0 = ay0.max(by0);
    let inter_x1 = ax1.min(bx1);
    let inter_y1 = ay1.min(by1);
    if inter_x1 < inter_x0 || inter_y1 < inter_y0 {
        return false;
    }
    let iw = inter_x1 - inter_x0 + 1;
    let ih = inter_y1 - inter_y0 + 1;
    let inter = iw as u64 * ih as u64;
    let area_a = (ax1 - ax0 + 1) as u64 * (ay1 - ay0 + 1) as u64;
    let area_b = (bx1 - bx0 + 1) as u64 * (by1 - by0 + 1) as u64;
    let union = area_a + area_b - inter;
    inter * 10 >= union * (tenths as u64)
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
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
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
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
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

pub fn layer0(device: &wgpu::Device, queue: &wgpu::Queue, img_view: &wgpu::TextureView, img_info: preprocessing::Imginfo, static_vals: &layer_static_values) -> (wgpu::Buffer, [u32; 4]) {
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
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(img_view) },
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
    img_view: &wgpu::TextureView,
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
    let inactive_avg = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer1_inactive_avg"),
        size: buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let zero = vec![0u8; buf_size as usize];
    queue.write_buffer(&out_mask, 0, &zero);
    queue.write_buffer(&inactive_avg, 0, &zero);

    let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("layer1_info"),
        contents: bytemuck::cast_slice(&info),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &static_vals.bindgroup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(img_view) },
            wgpu::BindGroupEntry { binding: 1, resource: info_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: l0_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: inactive_avg.as_entire_binding() },
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
        inactive_avg,
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
            inactive_avg: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("l2_pool_inactive_avg_empty"),
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
    let pooled_inactive_avg = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l2_pool_inactive_avg"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let zero = vec![0u8; out_size as usize];
    queue.write_buffer(&pooled_mask, 0, &zero);
    queue.write_buffer(&pooled_inactive_avg, 0, &zero);

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &static_vals.bindgroup_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: info_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: l1.mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: l1.inactive_avg.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pooled_mask.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pooled_inactive_avg.as_entire_binding() },
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
        inactive_avg: pooled_inactive_avg,
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

// ---- 메인 모델 조정 함수 ----

pub fn model(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img_view: wgpu::TextureView,
    img_info: preprocessing::Imginfo,
    static_vals: [&layer_static_values; 2], // [l0..l1]
    l2_static: &layer2_static_values,
    l3_edge_pipelines: &l3_edge_gpu::L3EdgeGpuPipelines,
    l3_edge_buffers: &mut Option<l3_edge_gpu::L3EdgeGpuBuffers>,
    l4_pipelines: &l4_gpu::L4ExpandPipelines,
    l4_buffers: &mut Option<l4_gpu::L4ExpandBuffers>,
    src_path: &Path,
) {
    let total_start = Instant::now();
    let mut l3_edge_e = std::time::Duration::from_secs(0);
    let mut l4_e = std::time::Duration::from_secs(0);

    // ---- Layer0 ----
    let l0_s = Instant::now();
    let (l0_dirs, info) = layer0(device, queue, &img_view, img_info, static_vals[0]);
    let l0_e = l0_s.elapsed();

    // ---- Layer1 ----
    let l1_s = Instant::now();
    let l1_out = layer1(device, queue, &img_view, &l0_dirs, info, static_vals[1], src_path);
    let l1_e = l1_s.elapsed();

    let l2_s = Instant::now();
    let l2_out = layer2(device, queue, &l1_out, info, l2_static);
    let l2_e = l2_s.elapsed();

    let height = info[0];
    let width = info[1];
    let mut pooled_cpu: Option<Vec<u32>> = None;
    if (config::get().save_layer2 || config::get().save_layer3_edge) && l2_out.out_w > 0 && l2_out.out_h > 0 {
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
            let inactive = readback_u32_buffer(device, queue, &l2_out.inactive_avg, out_size);
            let _ = visualization::save_layer2_inactive_avg_overlay(
                src_path,
                &inactive,
                l2_out.out_w as usize,
                l2_out.out_h as usize,
                "inactive",
            );
        }
        pooled_cpu = Some(pooled);
    }

    let need_l3_edge = config::get().save_layer3_edge || config::get().save_layer4 || config::get().log_timing;
    if need_l3_edge && l2_out.out_w >= 16 && l2_out.out_h >= 16 {
        let l3_edge_s = Instant::now();
        let aw = ((l2_out.out_w - 16) / 4) + 1;
        let ah = ((l2_out.out_h - 16) / 4) + 1;
        let bw = (aw + 3) / 4;
        let bh = (ah + 3) / 4;
        let bw2 = (bw + 1) / 2;
        let bh2 = (bh + 1) / 2;
        if aw > 0 && ah > 0 && bw > 0 && bh > 0 && bw2 > 0 && bh2 > 0 {
            let rebuild = l3_edge_buffers
                .as_ref()
                .map(|b| b.aw != aw || b.ah != ah || b.bw != bw || b.bh != bh || b.bw2 != bw2 || b.bh2 != bh2)
                .unwrap_or(true);
            if rebuild {
                *l3_edge_buffers = Some(l3_edge_gpu::ensure_l3_edge_buffers(device, aw, ah, bw, bh, bw2, bh2));
            }
            let l3_edge_bufs = l3_edge_buffers.as_ref().unwrap();

            let info_buf_data = [height, width];
            let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("l3_edge_info"),
                contents: bytemuck::cast_slice(&info_buf_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let l4_info_buf_data = [height, width, 0u32, 0u32];
            let l4_info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("layer4_info"),
                contents: bytemuck::cast_slice(&l4_info_buf_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("layer3_enc") });
            l3_edge_gpu::dispatch_l3_edge(
                device,
                &mut enc,
                l3_edge_pipelines,
                l3_edge_bufs,
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
            let block_slots = block_count * 40;
            let mut block_boxes: Vec<u32> = Vec::new();
            let mut block_valid: Vec<u32> = Vec::new();
            let mut l4_expanded: Vec<u32> = Vec::new();
            let mut l4_flags: Vec<u32> = Vec::new();
            let mut l4_merged: Vec<(u32, u32, u32, u32)> = Vec::new();

            let need_l4 = config::get().save_layer4 || config::get().log_timing;
            if need_l4 && block_slots > 0 {
                let l4_s = Instant::now();
                let n = block_slots as u32;
                let bin_size: u32 = 64;
                let bin_w = (l2_out.out_w + bin_size - 1) / bin_size;
                let bin_h = (l2_out.out_h + bin_size - 1) / bin_size;
                let bin_count = bin_w.saturating_mul(bin_h).max(1);
                let rebuild = l4_buffers
                    .as_ref()
                    .map(|b| b.n != n || b.bin_count != bin_count)
                    .unwrap_or(true);
                if rebuild {
                    *l4_buffers = Some(l4_gpu::ensure_l4_expand_buffers(device, n, bin_count));
                }
                let l4_bufs = l4_buffers.as_ref().unwrap();

                let params_data = [
                    config::get().l4_ring_pct,
                    config::get().l4_iou_tenths,
                    config::get().l4_gap_max,
                    config::get().l4_ov_pct,
                ];
                queue.write_buffer(&l4_bufs.params, 0, bytemuck::cast_slice(&params_data));

                let l4_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("layer4_expand_bg"),
                    layout: &l4_pipelines.expand_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: l4_info_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: l2_out.mask.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: l3_edge_bufs.block_boxes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: l3_edge_bufs.block_valid.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: l4_bufs.params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: l4_bufs.bin_counts.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: l4_bufs.bin_items.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: l4_bufs.expanded_boxes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: l4_bufs.expanded_flags.as_entire_binding() },
                    ],
                });

                let mut l4_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("layer4_enc") });
                {
                    let mut pass = l4_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("layer4_clear_bins"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&l4_pipelines.clear_bins);
                    pass.set_bind_group(0, &l4_bg, &[]);
                    pass.dispatch_workgroups((bin_count + 255) / 256, 1, 1);
                }
                {
                    let mut pass = l4_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("layer4_insert_bins"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&l4_pipelines.insert_bins);
                    pass.set_bind_group(0, &l4_bg, &[]);
                    pass.dispatch_workgroups((n + 255) / 256, 1, 1);
                }
                {
                    let mut pass = l4_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("layer4_expand_once"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&l4_pipelines.expand);
                    pass.set_bind_group(0, &l4_bg, &[]);
                    pass.dispatch_workgroups((n + 255) / 256, 1, 1);
                }
                queue.submit([l4_enc.finish()]);

                if config::get().save_layer4 {
                    let boxes_bytes = (n as u64) * 2 * 4;
                    let flags_bytes = (n as u64) * 4;
                    l4_expanded = readback_u32_buffer(device, queue, &l4_bufs.expanded_boxes, boxes_bytes);
                    l4_flags = readback_u32_buffer(device, queue, &l4_bufs.expanded_flags, flags_bytes);

                    let mut valid_boxes: Vec<(u32, u32, u32, u32)> = Vec::new();
                    for i in 0..(n as usize) {
                        if i >= l4_flags.len() {
                            break;
                        }
                        if (l4_flags[i] & (1u32 << 7u32)) == 0u32 {
                            continue;
                        }
                        let bi = i * 2;
                        if bi + 1 >= l4_expanded.len() {
                            break;
                        }
                        if let Some(b) = decode_bbox_pair(l4_expanded[bi], l4_expanded[bi + 1]) {
                            valid_boxes.push(b);
                        }
                    }

                    if !valid_boxes.is_empty() {
                        let mut dsu = DSU::new(valid_boxes.len());
                        let tenths = config::get().l4_iou_tenths.max(1);
                        for i in 0..valid_boxes.len() {
                            for j in (i + 1)..valid_boxes.len() {
                                if iou_ge_tenths(valid_boxes[i], valid_boxes[j], tenths) {
                                    dsu.union(i, j);
                                }
                            }
                        }

                        let mut merged: Vec<(u32, u32, u32, u32)> = Vec::new();
                        let mut root_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                        for (i, b) in valid_boxes.iter().enumerate() {
                            let root = dsu.find(i);
                            let entry = root_map.entry(root).or_insert_with(|| {
                                merged.push(*b);
                                merged.len() - 1
                            });
                            let cur = &mut merged[*entry];
                            cur.0 = cur.0.min(b.0);
                            cur.1 = cur.1.min(b.1);
                            cur.2 = cur.2.max(b.2);
                            cur.3 = cur.3.max(b.3);
                        }

                        merged.sort_by(|a, b| {
                            let area_a = (a.2 - a.0 + 1) as u64 * (a.3 - a.1 + 1) as u64;
                            let area_b = (b.2 - b.0 + 1) as u64 * (b.3 - b.1 + 1) as u64;
                            area_b.cmp(&area_a)
                                .then_with(|| a.1.cmp(&b.1))
                                .then_with(|| a.0.cmp(&b.0))
                        });
                        l4_merged = merged;
                    }

                    if block_slots > 0 && block_boxes.is_empty() {
                        let boxes_bytes = (block_slots * 2 * 4) as u64;
                        block_boxes = readback_u32_buffer(device, queue, &l3_edge_bufs.block_boxes, boxes_bytes);
                        let valid_bytes = (block_slots * 4) as u64;
                        block_valid = readback_u32_buffer(device, queue, &l3_edge_bufs.block_valid, valid_bytes);
                    }

                    let _ = visualization::save_layer4_expanded_overlay(
                        src_path,
                        l2_out.out_w as usize,
                        l2_out.out_h as usize,
                        &block_boxes,
                        &block_valid,
                        &l4_expanded,
                        &l4_flags,
                    );
                    let _ = visualization::save_layer4_merged_overlay(
                        src_path,
                        l2_out.out_w as usize,
                        l2_out.out_h as usize,
                        &l4_merged,
                    );
                }
                l4_e = l4_s.elapsed();
            }

            if config::get().save_layer3_edge && block_slots > 0 && block_boxes.is_empty() {
                let boxes_bytes = (block_slots * 2 * 4) as u64;
                block_boxes = readback_u32_buffer(device, queue, &l3_edge_bufs.block_boxes, boxes_bytes);
                let valid_bytes = (block_slots * 4) as u64;
                block_valid = readback_u32_buffer(device, queue, &l3_edge_bufs.block_valid, valid_bytes);
            }

            if config::get().save_layer3_edge {
                let anchor_count = (aw * ah) as usize;
                if anchor_count > 0 {
                    let anchor_bbox_bytes = (anchor_count * 2 * 4) as u64;
                    let anchor_score_bytes = (anchor_count * 4) as u64;
                    let anchor_bbox = readback_u32_buffer(device, queue, &l3_edge_bufs.anchor_bbox, anchor_bbox_bytes);
                    let anchor_score = readback_u32_buffer(device, queue, &l3_edge_bufs.anchor_score, anchor_score_bytes);
                    let anchor_meta = readback_u32_buffer(device, queue, &l3_edge_bufs.anchor_meta, anchor_score_bytes);

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
                        let _ = visualization::save_layer3_pass2b_block_overlay(
                            src_path,
                            pooled,
                            l2_out.out_w as usize,
                            l2_out.out_h as usize,
                            &block_boxes,
                            &block_valid,
                            "layer3_pass2b",
                        );
                    }
                }
            }

            l3_edge_e = l3_edge_s.elapsed();
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
    // 필요하면 추가로 레이어 시간을 별도 로그로 남기세요.
    if config::get().log_timing {
        visualization::log_timing_block(
            src_path,
            info,
            l0_e,
            l1_e,
            l2_e,
            l3_edge_e,
            l4_e,
            total_start.elapsed(),
        );
    }

    // 필요하면 추가로 레이어 시간을 별도 로그로 남기세요.
}
