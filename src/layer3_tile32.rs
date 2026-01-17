use crate::config::AppConfig;
use crate::rle_ccl_gpu::OutBox;

pub const BIN_SIZE: u32 = 32;

pub struct L3Tile32Pipelines {
    pub layout: wgpu::BindGroupLayout,
    pub clear: wgpu::ComputePipeline,
    pub accumulate: wgpu::ComputePipeline,
    pub emit: wgpu::ComputePipeline,
}

pub struct L3Tile32Buffers {
    pub box_capacity: u32,
    pub bins_x: u32,
    pub bins_y: u32,
    pub k: u32,
    pub slots_count: u32,
    pub out_capacity: u32,

    pub slot_color: wgpu::Buffer,
    pub minx: wgpu::Buffer,
    pub miny: wgpu::Buffer,
    pub maxx: wgpu::Buffer,
    pub maxy: wgpu::Buffer,
    pub out_boxes: wgpu::Buffer,
    pub out_count: wgpu::Buffer,
    pub params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct L3Params {
    box_count: u32,
    img_w: u32,
    img_h: u32,
    bins_x: u32,
    bins_y: u32,
    k: u32,
    min_area: u32,
    overflow_mode: u32,
}

fn readback_u32_buffer(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, byte_len: u64) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_readback_staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l3_tile32_readback_enc") });
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

fn readback_out_boxes(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, count: u32) -> Vec<OutBox> {
    if count == 0 {
        return Vec::new();
    }
    let bytes = (count as u64) * 16;
    let raw = readback_u32_buffer(device, queue, src, bytes);
    raw.chunks_exact(4)
        .map(|c| OutBox { x0y0: c[0], x1y1: c[1], color565: c[2], flags: c[3] })
        .collect()
}

pub fn build_l3_tile32_pipelines(
    device: &wgpu::Device,
    clear_shader: wgpu::ShaderModule,
    accumulate_shader: wgpu::ShaderModule,
    emit_shader: wgpu::ShaderModule,
) -> L3Tile32Pipelines {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("l3_tile32_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("l3_tile32_pipeline_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let mk = |label: &str, module: &wgpu::ShaderModule| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    L3Tile32Pipelines {
        layout,
        clear: mk("l3_tile32_clear_pipeline", &clear_shader),
        accumulate: mk("l3_tile32_accumulate_pipeline", &accumulate_shader),
        emit: mk("l3_tile32_emit_pipeline", &emit_shader),
    }
}

pub fn ensure_l3_tile32_buffers(device: &wgpu::Device, box_capacity: u32, bins_x: u32, bins_y: u32, k: u32) -> L3Tile32Buffers {
    let box_capacity = box_capacity.max(1);
    let bins_x = bins_x.max(1);
    let bins_y = bins_y.max(1);
    let k = k.max(1);
    let bins_count = bins_x.saturating_mul(bins_y).max(1);
    let slots_count = bins_count.saturating_mul(k).max(1);
    let out_capacity = slots_count;

    let slot_color = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_slot_color"),
        size: (slots_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let minx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_minx"),
        size: (slots_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let miny = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_miny"),
        size: (slots_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let maxx = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_maxx"),
        size: (slots_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let maxy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_maxy"),
        size: (slots_count as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let out_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_out_boxes"),
        size: (out_capacity as u64) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_out_count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l3_tile32_params"),
        size: std::mem::size_of::<L3Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    L3Tile32Buffers {
        box_capacity,
        bins_x,
        bins_y,
        k,
        slots_count,
        out_capacity,
        slot_color,
        minx,
        miny,
        maxx,
        maxy,
        out_boxes,
        out_count,
        params,
    }
}

pub fn run_l3_tile32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &L3Tile32Pipelines,
    bufs: &L3Tile32Buffers,
    boxes_in: &wgpu::Buffer,
    box_count: u32,
    img_w: u32,
    img_h: u32,
    cfg: &AppConfig,
) -> Vec<OutBox> {
    if box_count == 0 {
        return Vec::new();
    }
    let params = L3Params {
        box_count: box_count.min(bufs.box_capacity),
        img_w: img_w.max(1),
        img_h: img_h.max(1),
        bins_x: bufs.bins_x,
        bins_y: bufs.bins_y,
        k: bufs.k,
        min_area: cfg.l3_min_area,
        overflow_mode: cfg.l3_overflow_mode,
    };
    queue.write_buffer(&bufs.params, 0, bytemuck::bytes_of(&params));

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("l3_tile32_bg"),
        layout: &pipelines.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: boxes_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs.out_boxes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs.out_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs.params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs.slot_color.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs.minx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs.miny.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs.maxx.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs.maxy.as_entire_binding() },
        ],
    });

    let dispatch_1d_256 = |count: u32| (count + 255) / 256;

    // clear
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l3_tile32_clear_enc") });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l3_tile32_clear_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.clear);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dispatch_1d_256(bufs.slots_count), 1, 1);
        drop(pass);
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::Wait);
    }

    // accumulate
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l3_tile32_accum_enc") });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l3_tile32_accum_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.accumulate);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dispatch_1d_256(params.box_count), 1, 1);
        drop(pass);
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::Wait);
    }

    // emit (single invocation deterministic)
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("l3_tile32_emit_enc") });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("l3_tile32_emit_pass"), timestamp_writes: None });
        pass.set_pipeline(&pipelines.emit);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::Wait);
    }

    let out_count = readback_u32_buffer(device, queue, &bufs.out_count, 4)
        .get(0)
        .copied()
        .unwrap_or(0)
        .min(bufs.out_capacity);
    readback_out_boxes(device, queue, &bufs.out_boxes, out_count)
}

