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

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    if (arrayLength(&out_count) == 0u) { return; }

    let bins_count = params.bins_x * params.bins_y;
    let k = max(1u, params.k);
    let slots_count = bins_count * k;

    var out_i: u32 = atomicLoad(&out_count[0]);
    for (var slot_idx = 0u; slot_idx < slots_count; slot_idx = slot_idx + 1u) {
        if (slot_idx >= arrayLength(&slot_color)) { break; }
        let c = atomicLoad(&slot_color[slot_idx]);
        if (c == UINT_MAX) { continue; }
        let minx = atomicLoad(&minx_buf[slot_idx]);
        let miny = atomicLoad(&miny_buf[slot_idx]);
        let maxx = atomicLoad(&maxx_buf[slot_idx]);
        let maxy = atomicLoad(&maxy_buf[slot_idx]);
        if (maxx <= minx || maxy <= miny) { continue; }
        let w = maxx - minx;
        let h = maxy - miny;
        let area = w * h;
        if (params.min_area != 0u && area < params.min_area) { continue; }
        if (out_i >= arrayLength(&boxes_out)) { break; }
        let x0y0 = (miny << 16u) | (minx & 0xFFFFu);
        let x1y1 = (maxy << 16u) | (maxx & 0xFFFFu);
        // flags: 0 = normal (bin merge), 1 = overflow passthrough
        boxes_out[out_i] = OutBox(x0y0, x1y1, c & 0xFFFFu, 0u);
        out_i = out_i + 1u;
    }
    atomicStore(&out_count[0], out_i);
}

