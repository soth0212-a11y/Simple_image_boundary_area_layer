// L3 stride-2 pooling + 8-neighbor connectivity on L2 active mask.
// Output: s_active (0/1) and conn8 (bitmask in lower 8 bits).

const WG_X: u32 = 16u;
const WG_Y: u32 = 16u;
const CONSERVATIVE_DIAGONALS: bool = false;

@group(0) @binding(0) var<storage, read> dims: array<u32>; // [in_w, in_h, out_w, out_h]
@group(0) @binding(1) var<storage, read> in_active: array<u32>; // L2 mask (bit1=ACTIVE)
@group(0) @binding(2) var<storage, read_write> s_active: array<u32>; // out_w*out_h (0/1)

fn get_active(idx: u32) -> u32 {
    if (idx >= arrayLength(&in_active)) { return 0u; }
    let v: u32 = in_active[idx];
    return (v >> 1u) & 1u;
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn l3_stride2_pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&dims) < 4u) { return; }
    let in_w: u32 = dims[0];
    let in_h: u32 = dims[1];
    let out_w: u32 = dims[2];
    let out_h: u32 = dims[3];
    if (in_w == 0u || in_h == 0u || out_w == 0u || out_h == 0u) { return; }

    let sx: u32 = gid.x;
    let sy: u32 = gid.y;
    if (sx >= out_w || sy >= out_h) { return; }

    let x0: u32 = sx * 2u;
    let y0: u32 = sy * 2u;
    var a: u32 = 0u;
    for (var oy: u32 = 0u; oy < 2u; oy = oy + 1u) {
        let iy: u32 = y0 + oy;
        if (iy >= in_h) { continue; }
        for (var ox: u32 = 0u; ox < 2u; ox = ox + 1u) {
            let ix: u32 = x0 + ox;
            if (ix >= in_w) { continue; }
            let idx: u32 = iy * in_w + ix;
            a = a | get_active(idx);
        }
    }
    let out_idx: u32 = sy * out_w + sx;
    if (out_idx < arrayLength(&s_active)) {
        s_active[out_idx] = a & 1u;
    }
}

@group(0) @binding(0) var<storage, read> dims_conn: array<u32>; // [in_w, in_h, out_w, out_h]
@group(0) @binding(1) var<storage, read> s_active_in: array<u32>; // out_w*out_h (0/1)
@group(0) @binding(2) var<storage, read_write> conn8: array<u32>; // out_w*out_h (bitmask)

fn s_active_at(x: i32, y: i32, out_w: u32, out_h: u32) -> u32 {
    if (x < 0 || y < 0) { return 0u; }
    let ux: u32 = u32(x);
    let uy: u32 = u32(y);
    if (ux >= out_w || uy >= out_h) { return 0u; }
    let idx: u32 = uy * out_w + ux;
    if (idx >= arrayLength(&s_active_in)) { return 0u; }
    return s_active_in[idx] & 1u;
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn l3_conn8(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&dims_conn) < 4u) { return; }
    let out_w: u32 = dims_conn[2];
    let out_h: u32 = dims_conn[3];
    if (out_w == 0u || out_h == 0u) { return; }

    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if (x >= out_w || y >= out_h) { return; }
    let idx: u32 = y * out_w + x;
    if (idx >= arrayLength(&conn8)) { return; }

    let center: u32 = s_active_at(i32(x), i32(y), out_w, out_h);
    if (center == 0u) {
        conn8[idx] = 0u;
        return;
    }

    let n: u32 = s_active_at(i32(x), i32(y) - 1, out_w, out_h);
    let ne: u32 = s_active_at(i32(x) + 1, i32(y) - 1, out_w, out_h);
    let e: u32 = s_active_at(i32(x) + 1, i32(y), out_w, out_h);
    let se: u32 = s_active_at(i32(x) + 1, i32(y) + 1, out_w, out_h);
    let s: u32 = s_active_at(i32(x), i32(y) + 1, out_w, out_h);
    let sw: u32 = s_active_at(i32(x) - 1, i32(y) + 1, out_w, out_h);
    let w: u32 = s_active_at(i32(x) - 1, i32(y), out_w, out_h);
    let nw: u32 = s_active_at(i32(x) - 1, i32(y) - 1, out_w, out_h);

    var mask: u32 = 0u;
    if (n != 0u) { mask = mask | (1u << 0u); }
    if (e != 0u) { mask = mask | (1u << 2u); }
    if (s != 0u) { mask = mask | (1u << 4u); }
    if (w != 0u) { mask = mask | (1u << 6u); }

    if (CONSERVATIVE_DIAGONALS) {
        if (ne != 0u && (n != 0u || e != 0u)) { mask = mask | (1u << 1u); }
        if (se != 0u && (s != 0u || e != 0u)) { mask = mask | (1u << 3u); }
        if (sw != 0u && (s != 0u || w != 0u)) { mask = mask | (1u << 5u); }
        if (nw != 0u && (n != 0u || w != 0u)) { mask = mask | (1u << 7u); }
    } else {
        if (ne != 0u) { mask = mask | (1u << 1u); }
        if (se != 0u) { mask = mask | (1u << 3u); }
        if (sw != 0u) { mask = mask | (1u << 5u); }
        if (nw != 0u) { mask = mask | (1u << 7u); }
    }

    conn8[idx] = mask;
}
