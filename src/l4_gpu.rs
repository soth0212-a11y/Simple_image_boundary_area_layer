use wgpu;

const KMAX: u32 = 16;

pub struct L4GpuBuffers {
    pub nmax: u32,
    pub grid_w: u32,
    pub grid_h: u32,
    pub cell_count: u32,
    pub flat_boxes: wgpu::Buffer,
    pub flat_count: wgpu::Buffer,
    pub label: wgpu::Buffer,
    pub cell_counts: wgpu::Buffer,
    pub cell_items: wgpu::Buffer,
    pub overflow_flags: wgpu::Buffer,
    pub out_boxes: wgpu::Buffer,
    pub out_valid: wgpu::Buffer,
    pub params: wgpu::Buffer,
}

pub fn ensure_l4_buffers(device: &wgpu::Device, nmax: u32, grid_w: u32, grid_h: u32) -> L4GpuBuffers {
    let safe_nmax = nmax.max(1);
    let cell_count = grid_w.saturating_mul(grid_h).max(1);

    let flat_boxes_bytes = safe_nmax as u64 * 2 * 4;
    let flat_count_bytes = 4u64;
    let label_bytes = safe_nmax as u64 * 4;
    let cell_counts_bytes = cell_count as u64 * 4;
    let cell_items_bytes = cell_count as u64 * KMAX as u64 * 4;
    let overflow_bytes = cell_count as u64 * 4;
    let out_boxes_bytes = safe_nmax as u64 * 2 * 4;
    let out_valid_bytes = safe_nmax as u64 * 4;
    let params_bytes = 8u64;

    let flat_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_flat_boxes"),
        size: flat_boxes_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let flat_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_flat_count"),
        size: flat_count_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let label = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_label"),
        size: label_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let cell_counts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_cell_counts"),
        size: cell_counts_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cell_items = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_cell_items"),
        size: cell_items_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let overflow_flags = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_overflow_flags"),
        size: overflow_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_boxes = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_out_boxes"),
        size: out_boxes_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_valid = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_out_valid"),
        size: out_valid_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("l4_params"),
        size: params_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    L4GpuBuffers {
        nmax: safe_nmax,
        grid_w,
        grid_h,
        cell_count,
        flat_boxes,
        flat_count,
        label,
        cell_counts,
        cell_items,
        overflow_flags,
        out_boxes,
        out_valid,
        params,
    }
}
