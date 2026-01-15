// L4 bbox expand from L3 stride2 outputs (4-direction only).

const WG_X: u32 = 16u;
const WG_Y: u32 = 16u;

struct Dims {
    w: u32,
    h: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> s_active: array<u32>;
@group(0) @binding(1) var<storage, read> conn8: array<u32>;
@group(0) @binding(2) var<storage, read_write> bbox0: array<u32>;
@group(0) @binding(3) var<storage, read_write> bbox1: array<u32>;
@group(0) @binding(4) var<storage, read_write> meta_out: array<u32>;
@group(0) @binding(5) var<uniform> dims: Dims;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

fn bit(v: u32, b: u32) -> u32 {
    return (v >> b) & 1u;
}

fn active_3x3(x: u32, y: u32, w: u32, h: u32) -> u32 {
    var found: u32 = 0u;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nx: i32 = i32(x) + dx;
            let ny: i32 = i32(y) + dy;
            if (nx < 0 || ny < 0) { continue; }
            let ux: u32 = u32(nx);
            let uy: u32 = u32(ny);
            if (ux >= w || uy >= h) { continue; }
            let nidx: u32 = idx2(ux, uy, w);
            if (nidx >= arrayLength(&s_active)) { continue; }
            if ((s_active[nidx] & 1u) != 0u) {
                found = 1u;
            }
        }
    }
    return found;
}

fn cell_active(x: i32, y: i32, w: u32, h: u32) -> u32 {
    if (x < 0 || y < 0) { return 0u; }
    let ux: u32 = u32(x);
    let uy: u32 = u32(y);
    if (ux >= w || uy >= h) { return 0u; }
    let idx: u32 = idx2(ux, uy, w);
    if (idx >= arrayLength(&s_active)) { return 0u; }
    return s_active[idx] & 1u;
}

fn dir_ok_n(x: u32, y: u32, w: u32, h: u32) -> u32 {
    let xa: i32 = i32(x);
    let ya: i32 = i32(y);
    let a = cell_active(xa - 1, ya - 1, w, h);
    let b = cell_active(xa, ya - 1, w, h);
    let c = cell_active(xa + 1, ya - 1, w, h);
    return select(0u, 1u, (a != 0u && b != 0u && c != 0u));
}

fn dir_ok_e(x: u32, y: u32, w: u32, h: u32) -> u32 {
    let xa: i32 = i32(x);
    let ya: i32 = i32(y);
    let a = cell_active(xa + 1, ya - 1, w, h);
    let b = cell_active(xa + 1, ya, w, h);
    let c = cell_active(xa + 1, ya + 1, w, h);
    return select(0u, 1u, (a != 0u && b != 0u && c != 0u));
}

fn dir_ok_s(x: u32, y: u32, w: u32, h: u32) -> u32 {
    let xa: i32 = i32(x);
    let ya: i32 = i32(y);
    let a = cell_active(xa - 1, ya + 1, w, h);
    let b = cell_active(xa, ya + 1, w, h);
    let c = cell_active(xa + 1, ya + 1, w, h);
    return select(0u, 1u, (a != 0u && b != 0u && c != 0u));
}

fn dir_ok_w(x: u32, y: u32, w: u32, h: u32) -> u32 {
    let xa: i32 = i32(x);
    let ya: i32 = i32(y);
    let a = cell_active(xa - 1, ya - 1, w, h);
    let b = cell_active(xa - 1, ya, w, h);
    let c = cell_active(xa - 1, ya + 1, w, h);
    return select(0u, 1u, (a != 0u && b != 0u && c != 0u));
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn l4_build_bboxes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w: u32 = dims.w;
    let h: u32 = dims.h;
    if (w == 0u || h == 0u) { return; }

    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if (x >= w || y >= h) { return; }
    let idx: u32 = idx2(x, y, w);
    if (idx >= arrayLength(&s_active)) { return; }

    let is_active: u32 = active_3x3(x, y, w, h);
    if (is_active == 0u) {
        if (idx < arrayLength(&bbox0)) { bbox0[idx] = 0u; }
        if (idx < arrayLength(&bbox1)) { bbox1[idx] = 0u; }
        if (idx < arrayLength(&meta_out)) { meta_out[idx] = 0u; }
        return;
    }

    let c: u32 = conn8[idx] & 0xFFu;
    let n_of_a: u32 = bit(c, 0u);
    let e_of_a: u32 = bit(c, 2u);
    let s_of_a: u32 = bit(c, 4u);
    let w_of_a: u32 = bit(c, 6u);

    if ((n_of_a | e_of_a | s_of_a | w_of_a) == 0u) {
        if (idx < arrayLength(&bbox0)) { bbox0[idx] = 0u; }
        if (idx < arrayLength(&bbox1)) { bbox1[idx] = 0u; }
        if (idx < arrayLength(&meta_out)) { meta_out[idx] = 0u; }
        return;
    }

    var x0: u32 = x;
    var y0: u32 = y;
    var x1: u32 = x + 1u;
    var y1: u32 = y + 1u;
    var expand: u32 = 0u;

    if (x + 1u < w) {
        let idx_e: u32 = idx2(x + 1u, y, w);
        let e_ok: u32 = dir_ok_e(x, y, w, h);
        if (e_ok != 0u) {
            let c_e: u32 = conn8[idx_e] & 0xFFu;
            let w_of_b: u32 = bit(c_e, 6u);
            if (e_of_a != 0u && w_of_b != 0u) {
                x1 = x1 + 1u;
                expand = expand | (1u << 1u);
            }
        }
    }
    if (x > 0u) {
        let idx_w: u32 = idx2(x - 1u, y, w);
        let w_ok: u32 = dir_ok_w(x, y, w, h);
        if (w_ok != 0u) {
            let c_w: u32 = conn8[idx_w] & 0xFFu;
            let e_of_b: u32 = bit(c_w, 2u);
            if (w_of_a != 0u && e_of_b != 0u) {
                x0 = x0 - 1u;
                expand = expand | (1u << 3u);
            }
        }
    }
    if (y + 1u < h) {
        let idx_s: u32 = idx2(x, y + 1u, w);
        let s_ok: u32 = dir_ok_s(x, y, w, h);
        if (s_ok != 0u) {
            let c_s: u32 = conn8[idx_s] & 0xFFu;
            let n_of_b: u32 = bit(c_s, 0u);
            if (s_of_a != 0u && n_of_b != 0u) {
                y1 = y1 + 1u;
                expand = expand | (1u << 2u);
            }
        }
    }
    if (y > 0u) {
        let idx_n: u32 = idx2(x, y - 1u, w);
        let n_ok: u32 = dir_ok_n(x, y, w, h);
        if (n_ok != 0u) {
            let c_n: u32 = conn8[idx_n] & 0xFFu;
            let s_of_b: u32 = bit(c_n, 4u);
            if (n_of_a != 0u && s_of_b != 0u) {
                y0 = y0 - 1u;
                expand = expand | (1u << 0u);
            }
        }
    }

    let p0: u32 = (x0 & 0xFFFFu) | ((y0 & 0xFFFFu) << 16u);
    let p1: u32 = (x1 & 0xFFFFu) | ((y1 & 0xFFFFu) << 16u);
    if (idx < arrayLength(&bbox0)) { bbox0[idx] = p0; }
    if (idx < arrayLength(&bbox1)) { bbox1[idx] = p1; }
    if (idx < arrayLength(&meta_out)) { meta_out[idx] = expand & 0xFu; }
}
