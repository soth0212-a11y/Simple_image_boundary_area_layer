struct Params {
    groups_count: u32,
    bins_x: u32,
    bins_y: u32,
    color_tol: u32,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> group_count: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> group_minx: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> group_miny: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> group_maxx: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> group_maxy: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> group_color: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;

fn bits_r(r_shift: u32) -> u32 { return 5u - min(r_shift, 5u); }
fn bits_g(g_shift: u32) -> u32 { return 6u - min(g_shift, 6u); }
fn bits_b(b_shift: u32) -> u32 { return 5u - min(b_shift, 5u); }

fn color_close_565(a: u32, b: u32, tol: u32) -> bool {
    let qa = a & 0xFFFFu;
    let qb = b & 0xFFFFu;
    let ar = i32((qa >> 11u) & 31u);
    let ag = i32((qa >> 5u) & 63u);
    let ab = i32(qa & 31u);
    let br = i32((qb >> 11u) & 31u);
    let bg = i32((qb >> 5u) & 63u);
    let bb = i32(qb & 31u);
    let t = i32(tol);
    return abs(ar - br) <= t && abs(ag - bg) <= t && abs(ab - bb) <= t;
}

fn clamp_u32(v: i32, lo: i32, hi: i32) -> u32 {
    return u32(max(lo, min(hi, v)));
}

fn try_expand(group_id: u32, n_group_id: u32, my_color: u32) {
    if (n_group_id >= arrayLength(&group_count)) { return; }
    if (atomicLoad(&group_count[n_group_id]) == 0u) { return; }
    let n_color = atomicLoad(&group_color[n_group_id]);
    if (n_color == 0xFFFFFFFFu) { return; }
    if (!color_close_565(my_color, n_color, params.color_tol)) { return; }

    let n_minx = atomicLoad(&group_minx[n_group_id]);
    let n_miny = atomicLoad(&group_miny[n_group_id]);
    let n_maxx = atomicLoad(&group_maxx[n_group_id]);
    let n_maxy = atomicLoad(&group_maxy[n_group_id]);
    atomicMin(&group_minx[group_id], n_minx);
    atomicMin(&group_miny[group_id], n_miny);
    atomicMax(&group_maxx[group_id], n_maxx);
    atomicMax(&group_maxy[group_id], n_maxy);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let group_id = gid.x;
    if (group_id >= params.groups_count) { return; }
    if (group_id >= arrayLength(&group_count)) { return; }
    let cnt = atomicLoad(&group_count[group_id]);
    if (cnt == 0u) { return; }
    let my_color = atomicLoad(&group_color[group_id]);
    if (my_color == 0xFFFFFFFFu) { return; }

    let bins_count = params.bins_x * params.bins_y;
    if (bins_count == 0u) { return; }
    let bk = group_id / bins_count;
    let bin_id = group_id - bk * bins_count;
    let bin_x = bin_id % params.bins_x;
    let bin_y = bin_id / params.bins_x;

    // Expand only to 4-neighborhood (N/E/S/W) of bin_id (1 step),
    // within the SAME bandKey (bk), and only when q565 is within color_tol.
    //
    // NOTE: This corresponds to "group id 상/하/좌/우 1칸" where group_id is
    // bk*bins_count + bin_id.
    if (bin_x > 0u) {
        let n_group_id = bk * bins_count + (bin_id - 1u);
        try_expand(group_id, n_group_id, my_color);
    }
    if (bin_x + 1u < params.bins_x) {
        let n_group_id = bk * bins_count + (bin_id + 1u);
        try_expand(group_id, n_group_id, my_color);
    }
    if (bin_y > 0u) {
        let n_group_id = bk * bins_count + (bin_id - params.bins_x);
        try_expand(group_id, n_group_id, my_color);
    }
    if (bin_y + 1u < params.bins_y) {
        let n_group_id = bk * bins_count + (bin_id + params.bins_x);
        try_expand(group_id, n_group_id, my_color);
    }
}
