struct Params {
    total_boxes: u32,
    bins_x: u32,
    bins_y: u32,
    bin_size: u32,
}

struct OutBox {
    x0y0: u32,
    x1y1: u32,
    color565: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> in_boxes: array<OutBox>;
@group(0) @binding(1) var<storage, read> bin_offset: array<u32>;
@group(0) @binding(2) var<storage, read_write> bin_cursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bin_items: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn clamp_bin(v: u32, maxv: u32) -> u32 {
    if (maxv == 0u) { return 0u; }
    return min(v, maxv - 1u);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_boxes) { return; }
    if (idx >= arrayLength(&in_boxes)) { return; }
    let b = in_boxes[idx];
    let minx = b.x0y0 & 0xFFFFu;
    let miny = b.x0y0 >> 16u;
    let maxx = b.x1y1 & 0xFFFFu;
    let maxy = b.x1y1 >> 16u;
    let cx = (minx + maxx) >> 1u;
    let cy = (miny + maxy) >> 1u;
    let bin_x = clamp_bin(cx / params.bin_size, params.bins_x);
    let bin_y = clamp_bin(cy / params.bin_size, params.bins_y);
    let bin_id = bin_y * params.bins_x + bin_x;
    if (bin_id >= arrayLength(&bin_cursor) || bin_id >= arrayLength(&bin_offset)) { return; }
    let pos = atomicAdd(&bin_cursor[bin_id], 1u);
    let out_idx = bin_offset[bin_id] + pos;
    if (out_idx < arrayLength(&bin_items)) {
        bin_items[out_idx] = idx;
    }
}
