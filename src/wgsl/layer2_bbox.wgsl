struct Params {
    w: u32,
    h: u32,
    edge_cell_th: u32,
    area_th: u32,
    max_boxes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Box {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

@group(0) @binding(0) var<storage, read> edge_label: array<u32>;
@group(0) @binding(1) var<storage, read_write> bbox_minx: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> bbox_miny: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> bbox_maxx: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> bbox_maxy: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> edge_count: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> boxes_out: array<Box>;
@group(0) @binding(7) var<storage, read_write> box_count: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;
const INF: i32 = 0x3fffffff;
const NEG_INF: i32 = -0x3fffffff;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

@compute @workgroup_size(16, 16, 1)
fn l2_bbox_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&bbox_minx)) { return; }
    atomicStore(&bbox_minx[idx], INF);
    atomicStore(&bbox_miny[idx], INF);
    atomicStore(&bbox_maxx[idx], NEG_INF);
    atomicStore(&bbox_maxy[idx], NEG_INF);
    atomicStore(&edge_count[idx], 0u);
    if (idx == 0u && arrayLength(&box_count) > 0u) {
        atomicStore(&box_count[0], 0u);
    }
}

@compute @workgroup_size(16, 16, 1)
fn l2_bbox_accum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&edge_label) || idx >= arrayLength(&bbox_minx)) { return; }
    let lab = edge_label[idx];
    if (lab == UINT_MAX) { return; }
    if (lab >= arrayLength(&bbox_minx)) { return; }
    let xi: i32 = i32(x);
    let yi: i32 = i32(y);
    atomicMin(&bbox_minx[lab], xi);
    atomicMin(&bbox_miny[lab], yi);
    atomicMax(&bbox_maxx[lab], xi);
    atomicMax(&bbox_maxy[lab], yi);
    atomicAdd(&edge_count[lab], 1u);
}

@compute @workgroup_size(16, 16, 1)
fn l2_bbox_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&edge_label)) { return; }
    let lab = edge_label[idx];
    if (lab != idx) { return; }
    if (idx >= arrayLength(&bbox_minx)) { return; }

    let minx = atomicLoad(&bbox_minx[idx]);
    let miny = atomicLoad(&bbox_miny[idx]);
    let maxx = atomicLoad(&bbox_maxx[idx]);
    let maxy = atomicLoad(&bbox_maxy[idx]);
    let count = atomicLoad(&edge_count[idx]);

    if (minx > maxx || miny > maxy) { return; }
    let dx = maxx - minx + 1;
    let dy = maxy - miny + 1;
    if (dx <= 0 || dy <= 0) { return; }
    let area = u32(dx * dy);
    if (count < params.edge_cell_th) { return; }
    if (area < params.area_th) { return; }

    if (arrayLength(&box_count) == 0u) { return; }
    let out_i = atomicAdd(&box_count[0], 1u);
    if (out_i >= params.max_boxes || out_i >= arrayLength(&boxes_out)) { return; }
    boxes_out[out_i] = Box(minx, miny, maxx, maxy);
}
