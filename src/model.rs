use std::path::Path;
use std::time::Instant;
use wgpu;
use wgpu::util::DeviceExt;

use crate::config;
use crate::preprocessing;
use crate::visualization; // 시각화 모듈 사용

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
    src_path: &Path,
    l3_e: &mut std::time::Duration,
    l4_e: &mut std::time::Duration,
) {
    const KERNEL: u32 = 2;
    const STRIDE: u32 = 1;

    let height = img_info[0];
    let width = img_info[1];
    if width == 0 || height == 0 {
        return;
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

    let pooled = readback_u32_buffer(device, queue, &pooled_mask, out_size);
    if config::get().save_layer2 {
        let _ = visualization::save_layer2_overlay(
            src_path,
            &pooled,
            out_w as usize,
            out_h as usize,
            width as usize,
            height as usize,
            "layer2",
        );
    }

    if config::get().save_layer3 || config::get().save_layer4 || config::get().log_timing {
        let l3_s = Instant::now();
        let prefix = build_prefix_sum(&pooled, out_w, out_h);
        let candidates = generate_candidates(&prefix, out_w, out_h, 16, 4, 10);
        let kept = nms(candidates.clone());
        *l3_e = l3_s.elapsed();
        let l4_s = Instant::now();
        let l4 = l4_union_merge(&kept);
        *l4_e = l4_s.elapsed();

        if config::get().save_layer3 {
            let pre_boxes: Vec<(u32, u32, u32, u32)> = candidates
                .iter()
                .map(|b| (b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1))
                .collect();
            let kept_boxes: Vec<(u32, u32, u32, u32)> = kept
                .iter()
                .map(|b| (b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1))
                .collect();
            let _ = visualization::save_layer3_boxes_overlay(
                src_path,
                &pooled,
                out_w as usize,
                out_h as usize,
                &pre_boxes,
                &kept_boxes,
                "l3",
            );
        }
        if config::get().save_layer4 {
            let l4_boxes: Vec<(u32, u32, u32, u32)> = l4
                .iter()
                .map(|b| (b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1))
                .collect();
            let _ = visualization::save_layer4_boxes_overlay(
                src_path,
                &pooled,
                out_w as usize,
                out_h as usize,
                &l4_boxes,
                "l4",
            );
        }
    }

}

fn build_prefix_sum(active_mask: &[u32], w: u32, h: u32) -> Vec<u64> {
    let pw = w as usize + 1;
    let ph = h as usize + 1;
    let mut prefix = vec![0u64; pw * ph];
    for y in 0..h as usize {
        let row_base = (y + 1) * pw;
        let prev_base = y * pw;
        let mut row_sum: u64 = 0;
        for x in 0..w as usize {
            let idx = y * w as usize + x;
            let active = ((active_mask[idx] >> 1) & 1) as u64;
            row_sum += active;
            prefix[row_base + x + 1] = prefix[prev_base + x + 1] + row_sum;
        }
    }
    prefix
}

fn score_box(prefix: &[u64], w: u32, b: BBox) -> u64 {
    let x0 = b.x0 as usize;
    let y0 = b.y0 as usize;
    let x1 = b.x1 as usize;
    let y1 = b.y1 as usize;
    let pw = w as usize + 1;
    let a = prefix[y1 * pw + x1];
    let b0 = prefix[y0 * pw + x1];
    let c = prefix[y1 * pw + x0];
    let d = prefix[y0 * pw + x0];
    a - b0 - c + d
}

fn generate_candidates(
    prefix: &[u64],
    w: u32,
    h: u32,
    window: u32,
    stride: u32,
    threshold_pct: u32,
) -> Vec<BBoxScore> {
    let mut out = Vec::new();
    if w == 0 || h == 0 || window == 0 {
        return out;
    }
    let area = (window as u64) * (window as u64);
    let thresh = (threshold_pct as u64) * area;
    let mut y: u32 = 0;
    while y + window <= h {
        let mut x: u32 = 0;
        while x + window <= w {
            let b = BBox { x0: x, y0: y, x1: x + window, y1: y + window };
            let score = score_box(prefix, w, b);
            if score * 100 >= thresh {
                out.push(BBoxScore { bbox: b, score, area });
            }
            x += stride;
        }
        y += stride;
    }
    out
}

fn iou_ge_10(a: BBox, b: BBox) -> bool {
    let ax0 = a.x0;
    let ay0 = a.y0;
    let ax1 = a.x1;
    let ay1 = a.y1;
    let bx0 = b.x0;
    let by0 = b.y0;
    let bx1 = b.x1;
    let by1 = b.y1;
    let inter_x0 = ax0.max(bx0);
    let inter_y0 = ay0.max(by0);
    let inter_x1 = ax1.min(bx1);
    let inter_y1 = ay1.min(by1);
    if inter_x1 <= inter_x0 || inter_y1 <= inter_y0 {
        return false;
    }
    let inter_w = (inter_x1 - inter_x0) as u64;
    let inter_h = (inter_y1 - inter_y0) as u64;
    let inter_area = inter_w * inter_h;
    let area_a = ((ax1 - ax0) as u64) * ((ay1 - ay0) as u64);
    let area_b = ((bx1 - bx0) as u64) * ((by1 - by0) as u64);
    let union_area = area_a + area_b - inter_area;
    inter_area * 10 >= union_area
}

fn nms(mut boxes: Vec<BBoxScore>) -> Vec<BBoxScore> {
    boxes.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| b.area.cmp(&a.area)));
    let mut kept: Vec<BBoxScore> = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }
        let bi = boxes[i];
        kept.push(bi);
        for j in (i + 1)..boxes.len() {
            if suppressed[j] {
                continue;
            }
            let bj = boxes[j];
            if iou_ge_10(bi.bbox, bj.bbox) {
                suppressed[j] = true;
            }
        }
    }
    kept
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

fn l4_union_merge(boxes: &[BBoxScore]) -> Vec<BBoxScore> {
    let n = boxes.len();
    if n == 0 {
        return Vec::new();
    }
    let mut dsu = DSU::new(n);
    let radius_sq: u64 = 16u64 * 16u64;
    for i in 0..n {
        for j in (i + 1)..n {
            if center_distance_sq(boxes[i].bbox, boxes[j].bbox) <= radius_sq {
                dsu.union(i, j);
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
        });
        entry.bbox.x0 = entry.bbox.x0.min(b.bbox.x0);
        entry.bbox.y0 = entry.bbox.y0.min(b.bbox.y0);
        entry.bbox.x1 = entry.bbox.x1.max(b.bbox.x1);
        entry.bbox.y1 = entry.bbox.y1.max(b.bbox.y1);
        entry.score = entry.score.saturating_add(b.score);
    }
    let mut out: Vec<BBoxScore> = Vec::new();
    for (_root, mut v) in agg {
        let w = (v.bbox.x1.saturating_sub(v.bbox.x0)) as u64;
        let h = (v.bbox.y1.saturating_sub(v.bbox.y0)) as u64;
        v.area = w * h;
        out.push(v);
    }
    out
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
    layer2(device, queue, &l1_out, info, l2_static, src_path, &mut l3_e, &mut l4_e);
    let l2_e = l2_s.elapsed();

    let height = info[0];
    let width = info[1];
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
