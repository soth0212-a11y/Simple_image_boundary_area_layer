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

const BIN_SIZE: u32 = 32u;
const UINT_MAX: u32 = 0xFFFFFFFFu;

fn unpack_box(b: OutBox) -> vec4<u32> {
    let x0 = b.x0y0 & 0xFFFFu;
    let y0 = b.x0y0 >> 16u;
    let x1 = b.x1y1 & 0xFFFFu;
    let y1 = b.x1y1 >> 16u;
    return vec4<u32>(x0, y0, x1, y1);
}

fn clamp_u32(v: u32, lo: u32, hi: u32) -> u32 {
    return min(max(v, lo), hi);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.box_count) { return; }
    if (i >= arrayLength(&boxes_in)) { return; }
    let b = boxes_in[i];
    let bb = unpack_box(b);
    let x0 = bb.x;
    let y0 = bb.y;
    let x1 = bb.z;
    let y1 = bb.w;
    if (x1 <= x0 || y1 <= y0) { return; }
    let c = b.color565 & 0xFFFFu;

    let cx = (x0 + x1) / 2u;
    let cy = (y0 + y1) / 2u;
    var bx = cx / BIN_SIZE;
    var by = cy / BIN_SIZE;
    if (params.bins_x == 0u || params.bins_y == 0u) { return; }
    bx = clamp_u32(bx, 0u, params.bins_x - 1u);
    by = clamp_u32(by, 0u, params.bins_y - 1u);
    let bin_id = by * params.bins_x + bx;

    let k = max(1u, params.k);
    let base = bin_id * k;
    for (var slot = 0u; slot < k; slot = slot + 1u) {
        let idx = base + slot;
        if (idx >= arrayLength(&slot_color)) { return; }
        let existing = atomicLoad(&slot_color[idx]);
        if (existing == c) {
            atomicMin(&minx_buf[idx], x0);
            atomicMin(&miny_buf[idx], y0);
            atomicMax(&maxx_buf[idx], x1);
            atomicMax(&maxy_buf[idx], y1);
            return;
        }
        if (existing == UINT_MAX) {
            let res = atomicCompareExchangeWeak(&slot_color[idx], UINT_MAX, c);
            if (res.exchanged) {
                atomicStore(&minx_buf[idx], x0);
                atomicStore(&miny_buf[idx], y0);
                atomicStore(&maxx_buf[idx], x1);
                atomicStore(&maxy_buf[idx], y1);
                return;
            }
        }
    }

    // overflow
    if (params.overflow_mode == 1u) {
        if (arrayLength(&out_count) == 0u) { return; }
        let out_idx = atomicAdd(&out_count[0], 1u);
        if (out_idx < arrayLength(&boxes_out)) {
            // mark overflow fallback in flags=1
            boxes_out[out_idx] = OutBox(b.x0y0, b.x1y1, c, 1u);
        }
    }
}

