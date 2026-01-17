struct OutBox {
    x0y0: u32,
    x1y1: u32,
    color565: u32,
    flags: u32,
}

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

@group(0) @binding(0) var<storage, read> boxes_in: array<OutBox>;
@group(0) @binding(1) var<storage, read_write> boxes_out: array<OutBox>;
@group(0) @binding(2) var<storage, read_write> out_count: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: L3Params;
@group(0) @binding(4) var<storage, read_write> slot_color: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> minx_buf: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> miny_buf: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> maxx_buf: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> maxy_buf: array<atomic<u32>>;

const UINT_MAX: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_idx = gid.x;
    let bins_count = params.bins_x * params.bins_y;
    let slots_count = bins_count * max(1u, params.k);
    if (slot_idx >= slots_count) {
        return;
    }
    if (slot_idx < arrayLength(&slot_color)) {
        atomicStore(&slot_color[slot_idx], UINT_MAX);
    }
    if (slot_idx < arrayLength(&minx_buf)) {
        atomicStore(&minx_buf[slot_idx], UINT_MAX);
    }
    if (slot_idx < arrayLength(&miny_buf)) {
        atomicStore(&miny_buf[slot_idx], UINT_MAX);
    }
    if (slot_idx < arrayLength(&maxx_buf)) {
        atomicStore(&maxx_buf[slot_idx], 0u);
    }
    if (slot_idx < arrayLength(&maxy_buf)) {
        atomicStore(&maxy_buf[slot_idx], 0u);
    }
    if (slot_idx == 0u) {
        if (arrayLength(&out_count) != 0u) {
            atomicStore(&out_count[0], 0u);
        }
    }
}

