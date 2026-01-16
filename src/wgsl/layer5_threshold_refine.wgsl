// L5 threshold-based ROI refine + CCL + bbox accum (GPU-only).

const WG_TILE_X: u32 = 8u;
const WG_TILE_Y: u32 = 8u;
const WG_CELL_X: u32 = 16u;
const WG_CELL_Y: u32 = 16u;

const WIN: u32 = 2u;
const STR: u32 = 1u;
const MARGIN: u32 = 8u;
const UINT_MAX: u32 = 0xFFFFFFFFu;

struct Params0 {
    w: u32,
    h: u32,
    tile_w: u32,
    tile_h: u32,
}

struct Params1 {
    threshold: u32,
    iters: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> s_active: array<u32>;
@group(0) @binding(1) var<storage, read> conn8: array<u32>;
@group(0) @binding(2) var<storage, read_write> score_map: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_keep: array<u32>;
@group(0) @binding(4) var<storage, read_write> roi_mask: array<u32>;
@group(0) @binding(5) var<storage, read_write> label_a: array<u32>;
@group(0) @binding(6) var<storage, read_write> label_b: array<u32>;
@group(0) @binding(7) var<storage, read_write> acc_minx: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> acc_miny: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> acc_maxx: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> acc_maxy: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> acc_count: array<atomic<u32>>;
@group(0) @binding(12) var<uniform> params0: Params0;
@group(0) @binding(13) var<uniform> params1: Params1;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

fn conn4_deg(idx: u32) -> u32 {
    if (idx >= arrayLength(&conn8)) { return 0u; }
    let c = conn8[idx] & 0xFFu;
    let n = (c >> 0u) & 1u;
    let e = (c >> 2u) & 1u;
    let s = (c >> 4u) & 1u;
    let w = (c >> 6u) & 1u;
    return n + e + s + w;
}

@compute @workgroup_size(WG_TILE_X, WG_TILE_Y, 1)
fn l5_score_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx = gid.x;
    let ty = gid.y;
    if (tx >= params0.tile_w || ty >= params0.tile_h) { return; }
    let base_x = tx * STR;
    let base_y = ty * STR;
    var active_sum: u32 = 0u;
    var edge_sum: u32 = 0u;
    for (var oy: u32 = 0u; oy < WIN; oy = oy + 1u) {
        for (var ox: u32 = 0u; ox < WIN; ox = ox + 1u) {
            let gx = base_x + ox;
            let gy = base_y + oy;
            if (gx >= params0.w || gy >= params0.h) { continue; }
            let gidx = idx2(gx, gy, params0.w);
            if (gidx >= arrayLength(&s_active)) { continue; }
            let act = s_active[gidx] & 1u;
            active_sum = active_sum + act;
            if (act != 0u) {
                edge_sum = edge_sum + conn4_deg(gidx);
            }
        }
    }
    let score = active_sum + edge_sum;
    let tidx = idx2(tx, ty, params0.tile_w);
    if (tidx < arrayLength(&score_map)) {
        score_map[tidx] = score;
    }
}

@compute @workgroup_size(WG_TILE_X, WG_TILE_Y, 1)
fn l5_threshold_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx = gid.x;
    let ty = gid.y;
    if (tx >= params0.tile_w || ty >= params0.tile_h) { return; }
    let tidx = idx2(tx, ty, params0.tile_w);
    if (tidx >= arrayLength(&score_map) || tidx >= arrayLength(&tile_keep)) { return; }
    let score = score_map[tidx];
    tile_keep[tidx] = select(0u, 1u, score >= params1.threshold);
}

@compute @workgroup_size(WG_CELL_X, WG_CELL_Y, 1)
fn l5_upsample_roi_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params0.w || y >= params0.h) { return; }
    let max_tx = params0.tile_w - 1u;
    let max_ty = params0.tile_h - 1u;
    let span = (WIN - 1u) + MARGIN;
    let min_x = select(0u, x - span, x >= span);
    let max_x = x + MARGIN;
    let min_y = select(0u, y - span, y >= span);
    let max_y = y + MARGIN;
    var tx0 = min_x / STR;
    var ty0 = min_y / STR;
    var tx1 = max_x / STR;
    var ty1 = max_y / STR;
    if (tx1 > max_tx) { tx1 = max_tx; }
    if (ty1 > max_ty) { ty1 = max_ty; }
    var keep: u32 = 0u;
    for (var ty: u32 = ty0; ty <= ty1; ty = ty + 1u) {
        for (var tx: u32 = tx0; tx <= tx1; tx = tx + 1u) {
            let tidx = idx2(tx, ty, params0.tile_w);
            if (tidx >= arrayLength(&tile_keep)) { continue; }
            if (tile_keep[tidx] != 0u) {
                keep = 1u;
            }
        }
    }
    let idx = idx2(x, y, params0.w);
    if (idx < arrayLength(&roi_mask)) {
        roi_mask[idx] = keep;
    }
}

@compute @workgroup_size(WG_CELL_X, WG_CELL_Y, 1)
fn l5_init_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params0.w || y >= params0.h) { return; }
    let idx = idx2(x, y, params0.w);
    if (idx >= arrayLength(&label_a)) { return; }
    let masked = (idx < arrayLength(&roi_mask)) && (roi_mask[idx] != 0u) && (idx < arrayLength(&s_active)) && ((s_active[idx] & 1u) != 0u);
    label_a[idx] = select(UINT_MAX, idx, masked);
}

fn is_connected(idx: u32, nidx: u32, dir: u32) -> bool {
    if (idx >= arrayLength(&conn8) || nidx >= arrayLength(&conn8)) { return false; }
    let c0 = conn8[idx] & 0xFFu;
    let c1 = conn8[nidx] & 0xFFu;
    if (dir == 0u) { // N
        return ((c0 >> 0u) & 1u) != 0u && ((c1 >> 4u) & 1u) != 0u;
    }
    if (dir == 1u) { // E
        return ((c0 >> 2u) & 1u) != 0u && ((c1 >> 6u) & 1u) != 0u;
    }
    if (dir == 2u) { // S
        return ((c0 >> 4u) & 1u) != 0u && ((c1 >> 0u) & 1u) != 0u;
    }
    // W
    return ((c0 >> 6u) & 1u) != 0u && ((c1 >> 2u) & 1u) != 0u;
}

@compute @workgroup_size(WG_CELL_X, WG_CELL_Y, 1)
fn l5_propagate_labels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params0.w || y >= params0.h) { return; }
    let idx = idx2(x, y, params0.w);
    if (idx >= arrayLength(&label_a) || idx >= arrayLength(&label_b)) { return; }
    let lab = label_a[idx];
    if (lab == UINT_MAX) {
        label_b[idx] = UINT_MAX;
        return;
    }
    var best = lab;
    if (x > 0u) {
        let nidx = idx2(x - 1u, y, params0.w);
        let nlab = label_a[nidx];
        if (nlab != UINT_MAX && is_connected(idx, nidx, 3u)) {
            best = min(best, nlab);
        }
    }
    if (x + 1u < params0.w) {
        let nidx = idx2(x + 1u, y, params0.w);
        let nlab = label_a[nidx];
        if (nlab != UINT_MAX && is_connected(idx, nidx, 1u)) {
            best = min(best, nlab);
        }
    }
    if (y > 0u) {
        let nidx = idx2(x, y - 1u, params0.w);
        let nlab = label_a[nidx];
        if (nlab != UINT_MAX && is_connected(idx, nidx, 0u)) {
            best = min(best, nlab);
        }
    }
    if (y + 1u < params0.h) {
        let nidx = idx2(x, y + 1u, params0.w);
        let nlab = label_a[nidx];
        if (nlab != UINT_MAX && is_connected(idx, nidx, 2u)) {
            best = min(best, nlab);
        }
    }
    label_b[idx] = best;
}

@compute @workgroup_size(WG_CELL_X, WG_CELL_Y, 1)
fn l5_reset_accumulators(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params0.w || y >= params0.h) { return; }
    let idx = idx2(x, y, params0.w);
    if (idx >= arrayLength(&acc_minx)) { return; }
    atomicStore(&acc_minx[idx], UINT_MAX);
    atomicStore(&acc_miny[idx], UINT_MAX);
    atomicStore(&acc_maxx[idx], 0u);
    atomicStore(&acc_maxy[idx], 0u);
    atomicStore(&acc_count[idx], 0u);
}

@compute @workgroup_size(WG_CELL_X, WG_CELL_Y, 1)
fn l5_accumulate_bboxes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params0.w || y >= params0.h) { return; }
    let idx = idx2(x, y, params0.w);
    if (idx >= arrayLength(&label_a)) { return; }
    let lab = label_a[idx];
    if (lab == UINT_MAX) { return; }
    if (lab >= arrayLength(&acc_minx)) { return; }
    atomicMin(&acc_minx[lab], x);
    atomicMin(&acc_miny[lab], y);
    atomicMax(&acc_maxx[lab], x + 1u);
    atomicMax(&acc_maxy[lab], y + 1u);
    atomicAdd(&acc_count[lab], 1u);
}
