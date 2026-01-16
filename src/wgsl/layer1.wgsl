// L1 compute shader (2x2 reduce, per-channel mode from active/inactive pixels)
// - out_mask: u32 packed as (active_count << 16) | total_count
// - out_r/out_g/out_b: representative value (low 8 bits)

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>; // [height, width]
@group(0) @binding(1) var<storage, read> input_r: array<u32>;
@group(0) @binding(2) var<storage, read> input_g: array<u32>;
@group(0) @binding(3) var<storage, read> input_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_mask: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_r: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_g: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_b: array<u32>;

const BIT_N: u32 = 0u;
const BIT_NE: u32 = 1u;
const BIT_E: u32 = 2u;
const BIT_SE: u32 = 3u;
const BIT_S: u32 = 4u;
const BIT_SW: u32 = 5u;
const BIT_W: u32 = 6u;
const BIT_NW: u32 = 7u;

fn dir_mask(word: u32) -> u32 {
    return (word >> 1u) & 0xFFu;
}

fn val8(word: u32) -> u32 {
    return (word >> 9u) & 0xFFu;
}

fn compute_active_channel(x3: u32, y3: u32, grid3_w: u32, grid3_h: u32, chan: u32) -> bool {
    var neighbor_masks: array<u32, 9>;
    for (var mi: u32 = 0u; mi < 9u; mi = mi + 1u) {
        neighbor_masks[mi] = 0u;
    }

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        let ny: i32 = i32(y3) + dy;
        if (ny < 0 || ny >= i32(grid3_h)) { continue; }
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nx: i32 = i32(x3) + dx;
            if (nx < 0 || nx >= i32(grid3_w)) { continue; }
            let nidx: u32 = u32(ny) * grid3_w + u32(nx);
            var word: u32 = 0u;
            if (chan == 0u) {
                if (nidx < arrayLength(&input_r)) { word = input_r[nidx]; }
            } else if (chan == 1u) {
                if (nidx < arrayLength(&input_g)) { word = input_g[nidx]; }
            } else {
                if (nidx < arrayLength(&input_b)) { word = input_b[nidx]; }
            }
            let m_idx: u32 = u32((dy + 1) * 3 + (dx + 1));
            neighbor_masks[m_idx] = dir_mask(word);
        }
    }

    var cnt_n: u32 = 0u;
    var cnt_ne: u32 = 0u;
    var cnt_e: u32 = 0u;
    var cnt_se: u32 = 0u;
    var cnt_s: u32 = 0u;
    var cnt_sw: u32 = 0u;
    var cnt_w: u32 = 0u;
    var cnt_nw: u32 = 0u;

    for (var mi: u32 = 0u; mi < 9u; mi = mi + 1u) {
        let m: u32 = neighbor_masks[mi];
        cnt_n = cnt_n + select(0u, 1u, (m & (1u << BIT_N)) != 0u);
        cnt_ne = cnt_ne + select(0u, 1u, (m & (1u << BIT_NE)) != 0u);
        cnt_e = cnt_e + select(0u, 1u, (m & (1u << BIT_E)) != 0u);
        cnt_se = cnt_se + select(0u, 1u, (m & (1u << BIT_SE)) != 0u);
        cnt_s = cnt_s + select(0u, 1u, (m & (1u << BIT_S)) != 0u);
        cnt_sw = cnt_sw + select(0u, 1u, (m & (1u << BIT_SW)) != 0u);
        cnt_w = cnt_w + select(0u, 1u, (m & (1u << BIT_W)) != 0u);
        cnt_nw = cnt_nw + select(0u, 1u, (m & (1u << BIT_NW)) != 0u);
    }

    var inactive_dirs: u32 = 0u;
    var edge_dirs: u32 = 0u;
    if (cnt_n <= 2u) { inactive_dirs = inactive_dirs + 1u; } else if (cnt_n <= 6u) { edge_dirs = edge_dirs + 1u; }
    if (cnt_ne <= 2u) { inactive_dirs = inactive_dirs + 1u; } else if (cnt_ne <= 6u) { edge_dirs = edge_dirs + 1u; }
    if (cnt_e <= 2u) { inactive_dirs = inactive_dirs + 1u; } else if (cnt_e <= 6u) { edge_dirs = edge_dirs + 1u; }
    if (cnt_se <= 2u) { inactive_dirs = inactive_dirs + 1u; } else if (cnt_se <= 6u) { edge_dirs = edge_dirs + 1u; }
    if (cnt_s <= 2u) { inactive_dirs = inactive_dirs + 1u; } else if (cnt_s <= 6u) { edge_dirs = edge_dirs + 1u; }
    if (cnt_sw <= 2u) { inactive_dirs = inactive_dirs + 1u; } else if (cnt_sw <= 6u) { edge_dirs = edge_dirs + 1u; }
    if (cnt_w <= 2u) { inactive_dirs = inactive_dirs + 1u; } else if (cnt_w <= 6u) { edge_dirs = edge_dirs + 1u; }
    if (cnt_nw <= 2u) { inactive_dirs = inactive_dirs + 1u; } else if (cnt_nw <= 6u) { edge_dirs = edge_dirs + 1u; }

    return (inactive_dirs < 7u) && (edge_dirs >= 3u);
}

fn avg4(v0: u32, v1: u32, v2: u32, v3: u32, count: u32) -> u32 {
    if (count == 0u) { return 0u; }
    return (v0 + v1 + v2 + v3) / count;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&input_img_info) < 2u) { return; }
    let height: u32 = input_img_info[0];
    let width: u32 = input_img_info[1];
    if (width == 0u || height == 0u) { return; }

    let grid3_w: u32 = width;
    let grid3_h: u32 = height;
    let out_w: u32 = (grid3_w + 1u) / 2u;
    let out_h: u32 = (grid3_h + 1u) / 2u;
    if (gid.x >= out_w || gid.y >= out_h) { return; }

    let base_x: u32 = gid.x * 2u;
    let base_y: u32 = gid.y * 2u;
    var active_cnt: u32 = 0u;
    var total_cnt: u32 = 0u;

    var r_act_vals: array<u32, 4>;
    var r_in_vals: array<u32, 4>;
    var g_act_vals: array<u32, 4>;
    var g_in_vals: array<u32, 4>;
    var b_act_vals: array<u32, 4>;
    var b_in_vals: array<u32, 4>;
    var r_act_n: u32 = 0u;
    var r_in_n: u32 = 0u;
    var g_act_n: u32 = 0u;
    var g_in_n: u32 = 0u;
    var b_act_n: u32 = 0u;
    var b_in_n: u32 = 0u;
    var r_dir_or: u32 = 0u;
    var g_dir_or: u32 = 0u;
    var b_dir_or: u32 = 0u;

    for (var dy: u32 = 0u; dy < 2u; dy = dy + 1u) {
        let y3: u32 = base_y + dy;
        if (y3 >= grid3_h) { continue; }
        for (var dx: u32 = 0u; dx < 2u; dx = dx + 1u) {
            let x3: u32 = base_x + dx;
            if (x3 >= grid3_w) { continue; }
            total_cnt = total_cnt + 1u;
            let idx3: u32 = y3 * grid3_w + x3;
            let r_word = select(0u, input_r[idx3], idx3 < arrayLength(&input_r));
            let g_word = select(0u, input_g[idx3], idx3 < arrayLength(&input_g));
            let b_word = select(0u, input_b[idx3], idx3 < arrayLength(&input_b));

            let r_active = compute_active_channel(x3, y3, grid3_w, grid3_h, 0u);
            let g_active = compute_active_channel(x3, y3, grid3_w, grid3_h, 1u);
            let b_active = compute_active_channel(x3, y3, grid3_w, grid3_h, 2u);
            if (r_active || g_active || b_active) { active_cnt = active_cnt + 1u; }

            let rv = val8(r_word);
            let gv = val8(g_word);
            let bv = val8(b_word);

            if (r_active) { r_act_vals[r_act_n] = rv; r_act_n = r_act_n + 1u; r_dir_or = r_dir_or | dir_mask(r_word); } else { r_in_vals[r_in_n] = rv; r_in_n = r_in_n + 1u; }
            if (g_active) { g_act_vals[g_act_n] = gv; g_act_n = g_act_n + 1u; g_dir_or = g_dir_or | dir_mask(g_word); } else { g_in_vals[g_in_n] = gv; g_in_n = g_in_n + 1u; }
            if (b_active) { b_act_vals[b_act_n] = bv; b_act_n = b_act_n + 1u; b_dir_or = b_dir_or | dir_mask(b_word); } else { b_in_vals[b_in_n] = bv; b_in_n = b_in_n + 1u; }
        }
    }

    let out_idx: u32 = gid.y * out_w + gid.x;
    if (out_idx < arrayLength(&out_mask)) {
        out_mask[out_idx] = (active_cnt << 16u) | (total_cnt & 0xFFFFu);
    }
    if (out_idx < arrayLength(&out_r)) {
        let v = select(avg4(r_in_vals[0], r_in_vals[1], r_in_vals[2], r_in_vals[3], r_in_n),
                       avg4(r_act_vals[0], r_act_vals[1], r_act_vals[2], r_act_vals[3], r_act_n),
                       r_act_n > 0u);
        let dir = select(0u, r_dir_or, r_act_n > 0u);
        out_r[out_idx] = (v & 0xFFu) | ((dir & 0xFFu) << 8u);
    }
    if (out_idx < arrayLength(&out_g)) {
        let v = select(avg4(g_in_vals[0], g_in_vals[1], g_in_vals[2], g_in_vals[3], g_in_n),
                       avg4(g_act_vals[0], g_act_vals[1], g_act_vals[2], g_act_vals[3], g_act_n),
                       g_act_n > 0u);
        let dir = select(0u, g_dir_or, g_act_n > 0u);
        out_g[out_idx] = (v & 0xFFu) | ((dir & 0xFFu) << 8u);
    }
    if (out_idx < arrayLength(&out_b)) {
        let v = select(avg4(b_in_vals[0], b_in_vals[1], b_in_vals[2], b_in_vals[3], b_in_n),
                       avg4(b_act_vals[0], b_act_vals[1], b_act_vals[2], b_act_vals[3], b_act_n),
                       b_act_n > 0u);
        let dir = select(0u, b_dir_or, b_act_n > 0u);
        out_b[out_idx] = (v & 0xFFu) | ((dir & 0xFFu) << 8u);
    }
}
