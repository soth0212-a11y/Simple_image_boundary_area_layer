// L5 merge bboxes with direction gating + overlap/touch adjacency.

const WG_X: u32 = 16u;
const WG_Y: u32 = 16u;
const UINT_MAX: u32 = 0xFFFFFFFFu;

struct Dims {
    w: u32,
    h: u32,
    gap: u32,
    iterations: u32,
}

@group(0) @binding(0) var<storage, read> bbox0: array<u32>;
@group(0) @binding(1) var<storage, read> bbox1: array<u32>;
@group(0) @binding(2) var<storage, read> meta_in: array<u32>;
@group(0) @binding(3) var<storage, read> label_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> label_out: array<u32>;
@group(0) @binding(5) var<uniform> dims: Dims;
@group(0) @binding(6) var<storage, read_write> acc_minx: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> acc_miny: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> acc_maxx: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> acc_maxy: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> acc_count: array<atomic<u32>>;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

fn unpack_x(p: u32) -> u32 { return p & 0xFFFFu; }
fn unpack_y(p: u32) -> u32 { return (p >> 16u) & 0xFFFFu; }

fn valid_bbox(idx: u32) -> bool {
    if (idx >= arrayLength(&bbox0) || idx >= arrayLength(&bbox1)) { return false; }
    let p0 = bbox0[idx];
    let p1 = bbox1[idx];
    let x0 = unpack_x(p0);
    let y0 = unpack_y(p0);
    let x1 = unpack_x(p1);
    let y1 = unpack_y(p1);
    return (x1 > x0) && (y1 > y0);
}

fn allow_merge(idx_a: u32, idx_b: u32) -> bool {
    if (idx_a >= arrayLength(&meta_in) || idx_b >= arrayLength(&meta_in)) { return false; }
    let ma = meta_in[idx_a] & 0xFu;
    let mb = meta_in[idx_b] & 0xFu;
    return (ma != 0u) && (ma == mb);
}

fn rect_connected(idx_a: u32, idx_b: u32, gap: u32) -> bool {
    if (!valid_bbox(idx_a) || !valid_bbox(idx_b)) { return false; }
    let p0a = bbox0[idx_a];
    let p1a = bbox1[idx_a];
    let p0b = bbox0[idx_b];
    let p1b = bbox1[idx_b];
    let x0a = unpack_x(p0a);
    let y0a = unpack_y(p0a);
    let x1a = unpack_x(p1a);
    let y1a = unpack_y(p1a);
    let x0b = unpack_x(p0b);
    let y0b = unpack_y(p0b);
    let x1b = unpack_x(p1b);
    let y1b = unpack_y(p1b);
    let x1ag = x1a + gap;
    let y1ag = y1a + gap;
    let x1bg = x1b + gap;
    let y1bg = y1b + gap;
    return (x0a <= x1bg) && (x1ag >= x0b) && (y0a <= y1bg) && (y1ag >= y0b);
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn l5_init_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = dims.w;
    let h = dims.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&label_out)) { return; }
    if (valid_bbox(idx)) {
        label_out[idx] = idx;
    } else {
        label_out[idx] = UINT_MAX;
    }
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn l5_propagate_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = dims.w;
    let h = dims.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&label_in) || idx >= arrayLength(&label_out)) { return; }
    let lab = label_in[idx];
    if (lab == UINT_MAX) {
        label_out[idx] = UINT_MAX;
        return;
    }
    var best = lab;
    let gap = 0u;
    if (x > 0u) {
        let nidx = idx2(x - 1u, y, w);
        if (nidx < arrayLength(&label_in)) {
            let nlab = label_in[nidx];
            if (nlab != UINT_MAX && allow_merge(idx, nidx) && rect_connected(idx, nidx, gap)) {
                best = min(best, nlab);
            }
        }
    }
    if (x + 1u < w) {
        let nidx = idx2(x + 1u, y, w);
        if (nidx < arrayLength(&label_in)) {
            let nlab = label_in[nidx];
            if (nlab != UINT_MAX && allow_merge(idx, nidx) && rect_connected(idx, nidx, gap)) {
                best = min(best, nlab);
            }
        }
    }
    if (y > 0u) {
        let nidx = idx2(x, y - 1u, w);
        if (nidx < arrayLength(&label_in)) {
            let nlab = label_in[nidx];
            if (nlab != UINT_MAX && allow_merge(idx, nidx) && rect_connected(idx, nidx, gap)) {
                best = min(best, nlab);
            }
        }
    }
    if (y + 1u < h) {
        let nidx = idx2(x, y + 1u, w);
        if (nidx < arrayLength(&label_in)) {
            let nlab = label_in[nidx];
            if (nlab != UINT_MAX && allow_merge(idx, nidx) && rect_connected(idx, nidx, gap)) {
                best = min(best, nlab);
            }
        }
    }
    label_out[idx] = best;
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn l5_reset_accumulators(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = dims.w;
    let h = dims.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&acc_minx)) { return; }
    atomicStore(&acc_minx[idx], UINT_MAX);
    atomicStore(&acc_miny[idx], UINT_MAX);
    atomicStore(&acc_maxx[idx], 0u);
    atomicStore(&acc_maxy[idx], 0u);
    atomicStore(&acc_count[idx], 0u);
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn l5_accumulate_bboxes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = dims.w;
    let h = dims.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&label_in)) { return; }
    let lab = label_in[idx];
    if (lab == UINT_MAX) { return; }
    if (lab >= arrayLength(&acc_minx)) { return; }
    if (!valid_bbox(idx)) { return; }
    let p0 = bbox0[idx];
    let p1 = bbox1[idx];
    let x0 = unpack_x(p0);
    let y0 = unpack_y(p0);
    let x1 = unpack_x(p1);
    let y1 = unpack_y(p1);
    atomicMin(&acc_minx[lab], x0);
    atomicMin(&acc_miny[lab], y0);
    atomicMax(&acc_maxx[lab], x1);
    atomicMax(&acc_maxy[lab], y1);
    atomicAdd(&acc_count[lab], 1u);
}
