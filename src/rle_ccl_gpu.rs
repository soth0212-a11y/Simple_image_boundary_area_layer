use wgpu;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Segment {
    pub tl: u32,
    pub br: u32,
    pub color565: u32,
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

impl OutBox {
    #[inline]
    pub fn minx(&self) -> u32 {
        self.x0y0 & 0xFFFFu32
    }

    #[inline]
    pub fn miny(&self) -> u32 {
        self.x0y0 >> 16
    }

    #[inline]
    pub fn maxx(&self) -> u32 {
        self.x1y1 & 0xFFFFu32
    }

    #[inline]
    pub fn maxy(&self) -> u32 {
        self.x1y1 >> 16
    }

    #[inline]
    pub fn w(&self) -> u32 {
        let minx = self.minx();
        let maxx = self.maxx();
        if maxx > minx { maxx - minx } else { 0 }
    }

    #[inline]
    pub fn h(&self) -> u32 {
        let miny = self.miny();
        let maxy = self.maxy();
        if maxy > miny { maxy - miny } else { 0 }
    }

    #[inline]
    pub fn area(&self) -> u64 {
        (self.w() as u64) * (self.h() as u64)
    }

    #[inline]
    pub fn from_xy(minx: u32, miny: u32, maxx: u32, maxy: u32, color565: u32, flags: u32) -> Self {
        let x0y0 = (miny << 16) | (minx & 0xFFFFu32);
        let x1y1 = (maxy << 16) | (maxx & 0xFFFFu32);
        Self { x0y0, x1y1, color565, flags }
    }
}

pub struct L1RlePipelines {
    pub count_layout: wgpu::BindGroupLayout,
    pub count_pipeline: wgpu::ComputePipeline,
    pub emit_layout: wgpu::BindGroupLayout,
    pub emit_pipeline: wgpu::ComputePipeline,
}

pub struct L1RleBuffers {
    pub w: u32,
    pub h: u32,
    pub total_segments: u32,
    pub row_counts: wgpu::Buffer,
    pub row_offsets: wgpu::Buffer,
    pub segments: wgpu::Buffer,
    pub params_count: wgpu::Buffer,
    pub params_emit: wgpu::Buffer,
}

pub fn build_l1_rle_pipelines(
    device: &wgpu::Device,
    count_shader: wgpu::ShaderModule,
    emit_shader: wgpu::ShaderModule,
) -> L1RlePipelines {
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

    L1RlePipelines {
        count_layout,
        count_pipeline,
        emit_layout,
        emit_pipeline,
    }
}

pub fn ensure_l1_rle_buffers(device: &wgpu::Device, w: u32, h: u32, total_segments: u32) -> L1RleBuffers {
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

    L1RleBuffers {
        w,
        h,
        total_segments,
        row_counts,
        row_offsets,
        segments,
        params_count,
        params_emit,
    }
}
