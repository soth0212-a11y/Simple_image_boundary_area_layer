struct Params {
    total_boxes: u32,
    bins_x: u32,
    bins_y: u32,
    bin_size: u32,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
    _pad0: u32,
}

struct Segment {
    tl: u32,
    br: u32,
    color565: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> in_boxes: array<Segment>;
@group(0) @binding(1) var<storage, read_write> group_count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> group_minx: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> group_miny: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> group_maxx: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> group_maxy: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> group_color: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

fn clamp_bin(v: u32, maxv: u32) -> u32 {
    if (maxv == 0u) { return 0u; }
    return min(v, maxv - 1u);
}

fn bits_r(r_shift: u32) -> u32 { return 5u - min(r_shift, 5u); }
fn bits_g(g_shift: u32) -> u32 { return 6u - min(g_shift, 6u); }
fn bits_b(b_shift: u32) -> u32 { return 5u - min(b_shift, 5u); }

fn band_key_565(c: u32, r_shift: u32, g_shift: u32, b_shift: u32) -> u32 {
    let q = c & 0xFFFFu;
    let r5 = (q >> 11u) & 31u;
    let g6 = (q >> 5u) & 63u;
    let b5 = q & 31u;
    let rb = r5 >> min(r_shift, 4u);
    let gb = g6 >> min(g_shift, 5u);
    let bb = b5 >> min(b_shift, 4u);
    let bb_bits = bits_b(b_shift);
    let gb_bits = bits_g(g_shift);
    return (rb << (gb_bits + bb_bits)) | (gb << bb_bits) | bb;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_boxes) { return; }
    if (idx >= arrayLength(&in_boxes)) { return; }
    let b = in_boxes[idx];
    let minx = b.tl & 0xFFFFu;
    let miny = b.tl >> 16u;
    let maxx = b.br & 0xFFFFu;
    let maxy = b.br >> 16u;
    let cx = (minx + maxx) >> 1u;
    let cy = (miny + maxy) >> 1u;
    let bin_x = clamp_bin(cx / params.bin_size, params.bins_x);
    let bin_y = clamp_bin(cy / params.bin_size, params.bins_y);
    let bin_id = bin_y * params.bins_x + bin_x;

    let bk = band_key_565(b.color565, params.r_shift, params.g_shift, params.b_shift);
    let group_id = bk * (params.bins_x * params.bins_y) + bin_id;
    if (group_id >= arrayLength(&group_count)) { return; }

    atomicAdd(&group_count[group_id], 1u);
    atomicMin(&group_minx[group_id], minx);
    atomicMin(&group_miny[group_id], miny);
    atomicMax(&group_maxx[group_id], maxx);
    atomicMax(&group_maxy[group_id], maxy);
    atomicMin(&group_color[group_id], b.color565 & 0xFFFFu);
}

