// L1 compute shader (2x2 reduce)
// - gid.xy -> output cell (ox, oy) in half-resolution grid
// - out_mask: u32 packed as (active_count << 16) | total_count
// - 모든 연산은 정수(u32/i32)만 사용

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>; // [height, width]
@group(0) @binding(1) var<storage, read> layer0_masks: array<u32>;   // 1:1 u32 packed (R/G/B 8방향)
@group(0) @binding(2) var<storage, read_write> out_mask: array<u32>; // packed counts

const BIT_N: u32 = 0u;
const BIT_NE: u32 = 1u;
const BIT_E: u32 = 2u;
const BIT_SE: u32 = 3u;
const BIT_S: u32 = 4u;
const BIT_SW: u32 = 5u;
const BIT_W: u32 = 6u;
const BIT_NW: u32 = 7u;

fn compute_combined_active(x3: u32, y3: u32, grid3_w: u32, grid3_h: u32) -> bool {
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
            if (nidx >= arrayLength(&layer0_masks)) { continue; }
            let m_idx: u32 = u32((dy + 1) * 3 + (dx + 1));
            neighbor_masks[m_idx] = layer0_masks[nidx];
        }
    }

    // 8방향별로 누적 (채널별 bit count)
    var cnt_n_r: u32 = 0u;
    var cnt_ne_r: u32 = 0u;
    var cnt_e_r: u32 = 0u;
    var cnt_se_r: u32 = 0u;
    var cnt_s_r: u32 = 0u;
    var cnt_sw_r: u32 = 0u;
    var cnt_w_r: u32 = 0u;
    var cnt_nw_r: u32 = 0u;

    var cnt_n_g: u32 = 0u;
    var cnt_ne_g: u32 = 0u;
    var cnt_e_g: u32 = 0u;
    var cnt_se_g: u32 = 0u;
    var cnt_s_g: u32 = 0u;
    var cnt_sw_g: u32 = 0u;
    var cnt_w_g: u32 = 0u;
    var cnt_nw_g: u32 = 0u;

    var cnt_n_b: u32 = 0u;
    var cnt_ne_b: u32 = 0u;
    var cnt_e_b: u32 = 0u;
    var cnt_se_b: u32 = 0u;
    var cnt_s_b: u32 = 0u;
    var cnt_sw_b: u32 = 0u;
    var cnt_w_b: u32 = 0u;
    var cnt_nw_b: u32 = 0u;

    for (var mi: u32 = 0u; mi < 9u; mi = mi + 1u) {
        let m: u32 = neighbor_masks[mi];
        let mr: u32 = m & 0xFFu;
        let mg: u32 = (m >> 8u) & 0xFFu;
        let mb: u32 = (m >> 16u) & 0xFFu;

        cnt_n_r = cnt_n_r + select(0u, 1u, (mr & (1u << BIT_N)) != 0u);
        cnt_ne_r = cnt_ne_r + select(0u, 1u, (mr & (1u << BIT_NE)) != 0u);
        cnt_e_r = cnt_e_r + select(0u, 1u, (mr & (1u << BIT_E)) != 0u);
        cnt_se_r = cnt_se_r + select(0u, 1u, (mr & (1u << BIT_SE)) != 0u);
        cnt_s_r = cnt_s_r + select(0u, 1u, (mr & (1u << BIT_S)) != 0u);
        cnt_sw_r = cnt_sw_r + select(0u, 1u, (mr & (1u << BIT_SW)) != 0u);
        cnt_w_r = cnt_w_r + select(0u, 1u, (mr & (1u << BIT_W)) != 0u);
        cnt_nw_r = cnt_nw_r + select(0u, 1u, (mr & (1u << BIT_NW)) != 0u);

        cnt_n_g = cnt_n_g + select(0u, 1u, (mg & (1u << BIT_N)) != 0u);
        cnt_ne_g = cnt_ne_g + select(0u, 1u, (mg & (1u << BIT_NE)) != 0u);
        cnt_e_g = cnt_e_g + select(0u, 1u, (mg & (1u << BIT_E)) != 0u);
        cnt_se_g = cnt_se_g + select(0u, 1u, (mg & (1u << BIT_SE)) != 0u);
        cnt_s_g = cnt_s_g + select(0u, 1u, (mg & (1u << BIT_S)) != 0u);
        cnt_sw_g = cnt_sw_g + select(0u, 1u, (mg & (1u << BIT_SW)) != 0u);
        cnt_w_g = cnt_w_g + select(0u, 1u, (mg & (1u << BIT_W)) != 0u);
        cnt_nw_g = cnt_nw_g + select(0u, 1u, (mg & (1u << BIT_NW)) != 0u);

        cnt_n_b = cnt_n_b + select(0u, 1u, (mb & (1u << BIT_N)) != 0u);
        cnt_ne_b = cnt_ne_b + select(0u, 1u, (mb & (1u << BIT_NE)) != 0u);
        cnt_e_b = cnt_e_b + select(0u, 1u, (mb & (1u << BIT_E)) != 0u);
        cnt_se_b = cnt_se_b + select(0u, 1u, (mb & (1u << BIT_SE)) != 0u);
        cnt_s_b = cnt_s_b + select(0u, 1u, (mb & (1u << BIT_S)) != 0u);
        cnt_sw_b = cnt_sw_b + select(0u, 1u, (mb & (1u << BIT_SW)) != 0u);
        cnt_w_b = cnt_w_b + select(0u, 1u, (mb & (1u << BIT_W)) != 0u);
        cnt_nw_b = cnt_nw_b + select(0u, 1u, (mb & (1u << BIT_NW)) != 0u);
    }

    var inactive_dirs_r: u32 = 0u;
    var edge_dirs_r: u32 = 0u;
    if (cnt_n_r <= 2u) { inactive_dirs_r = inactive_dirs_r + 1u; } else if (cnt_n_r <= 6u) { edge_dirs_r = edge_dirs_r + 1u; }
    if (cnt_ne_r <= 2u) { inactive_dirs_r = inactive_dirs_r + 1u; } else if (cnt_ne_r <= 6u) { edge_dirs_r = edge_dirs_r + 1u; }
    if (cnt_e_r <= 2u) { inactive_dirs_r = inactive_dirs_r + 1u; } else if (cnt_e_r <= 6u) { edge_dirs_r = edge_dirs_r + 1u; }
    if (cnt_se_r <= 2u) { inactive_dirs_r = inactive_dirs_r + 1u; } else if (cnt_se_r <= 6u) { edge_dirs_r = edge_dirs_r + 1u; }
    if (cnt_s_r <= 2u) { inactive_dirs_r = inactive_dirs_r + 1u; } else if (cnt_s_r <= 6u) { edge_dirs_r = edge_dirs_r + 1u; }
    if (cnt_sw_r <= 2u) { inactive_dirs_r = inactive_dirs_r + 1u; } else if (cnt_sw_r <= 6u) { edge_dirs_r = edge_dirs_r + 1u; }
    if (cnt_w_r <= 2u) { inactive_dirs_r = inactive_dirs_r + 1u; } else if (cnt_w_r <= 6u) { edge_dirs_r = edge_dirs_r + 1u; }
    if (cnt_nw_r <= 2u) { inactive_dirs_r = inactive_dirs_r + 1u; } else if (cnt_nw_r <= 6u) { edge_dirs_r = edge_dirs_r + 1u; }

    var inactive_dirs_g: u32 = 0u;
    var edge_dirs_g: u32 = 0u;
    if (cnt_n_g <= 2u) { inactive_dirs_g = inactive_dirs_g + 1u; } else if (cnt_n_g <= 6u) { edge_dirs_g = edge_dirs_g + 1u; }
    if (cnt_ne_g <= 2u) { inactive_dirs_g = inactive_dirs_g + 1u; } else if (cnt_ne_g <= 6u) { edge_dirs_g = edge_dirs_g + 1u; }
    if (cnt_e_g <= 2u) { inactive_dirs_g = inactive_dirs_g + 1u; } else if (cnt_e_g <= 6u) { edge_dirs_g = edge_dirs_g + 1u; }
    if (cnt_se_g <= 2u) { inactive_dirs_g = inactive_dirs_g + 1u; } else if (cnt_se_g <= 6u) { edge_dirs_g = edge_dirs_g + 1u; }
    if (cnt_s_g <= 2u) { inactive_dirs_g = inactive_dirs_g + 1u; } else if (cnt_s_g <= 6u) { edge_dirs_g = edge_dirs_g + 1u; }
    if (cnt_sw_g <= 2u) { inactive_dirs_g = inactive_dirs_g + 1u; } else if (cnt_sw_g <= 6u) { edge_dirs_g = edge_dirs_g + 1u; }
    if (cnt_w_g <= 2u) { inactive_dirs_g = inactive_dirs_g + 1u; } else if (cnt_w_g <= 6u) { edge_dirs_g = edge_dirs_g + 1u; }
    if (cnt_nw_g <= 2u) { inactive_dirs_g = inactive_dirs_g + 1u; } else if (cnt_nw_g <= 6u) { edge_dirs_g = edge_dirs_g + 1u; }

    var inactive_dirs_b: u32 = 0u;
    var edge_dirs_b: u32 = 0u;
    if (cnt_n_b <= 2u) { inactive_dirs_b = inactive_dirs_b + 1u; } else if (cnt_n_b <= 6u) { edge_dirs_b = edge_dirs_b + 1u; }
    if (cnt_ne_b <= 2u) { inactive_dirs_b = inactive_dirs_b + 1u; } else if (cnt_ne_b <= 6u) { edge_dirs_b = edge_dirs_b + 1u; }
    if (cnt_e_b <= 2u) { inactive_dirs_b = inactive_dirs_b + 1u; } else if (cnt_e_b <= 6u) { edge_dirs_b = edge_dirs_b + 1u; }
    if (cnt_se_b <= 2u) { inactive_dirs_b = inactive_dirs_b + 1u; } else if (cnt_se_b <= 6u) { edge_dirs_b = edge_dirs_b + 1u; }
    if (cnt_s_b <= 2u) { inactive_dirs_b = inactive_dirs_b + 1u; } else if (cnt_s_b <= 6u) { edge_dirs_b = edge_dirs_b + 1u; }
    if (cnt_sw_b <= 2u) { inactive_dirs_b = inactive_dirs_b + 1u; } else if (cnt_sw_b <= 6u) { edge_dirs_b = edge_dirs_b + 1u; }
    if (cnt_w_b <= 2u) { inactive_dirs_b = inactive_dirs_b + 1u; } else if (cnt_w_b <= 6u) { edge_dirs_b = edge_dirs_b + 1u; }
    if (cnt_nw_b <= 2u) { inactive_dirs_b = inactive_dirs_b + 1u; } else if (cnt_nw_b <= 6u) { edge_dirs_b = edge_dirs_b + 1u; }

    let r_active: bool = (inactive_dirs_r < 7u) && (edge_dirs_r >= 3u);
    let g_active: bool = (inactive_dirs_g < 7u) && (edge_dirs_g >= 3u);
    let b_active: bool = (inactive_dirs_b < 7u) && (edge_dirs_b >= 3u);
    let combined_active: bool = r_active || g_active || b_active;
    return combined_active;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&input_img_info) < 2u) {
        return;
    }
    let height: u32 = input_img_info[0];
    let width: u32 = input_img_info[1];
    if (width == 0u || height == 0u) {
        return;
    }

    let grid3_w: u32 = width;
    let grid3_h: u32 = height;
    if (grid3_w == 0u || grid3_h == 0u) {
        return;
    }

    let out_w: u32 = (grid3_w + 1u) / 2u;
    let out_h: u32 = (grid3_h + 1u) / 2u;
    if (gid.x >= out_w || gid.y >= out_h) {
        return;
    }

    let base_x: u32 = gid.x * 2u;
    let base_y: u32 = gid.y * 2u;
    var active_cnt: u32 = 0u;
    var total_cnt: u32 = 0u;
    for (var dy: u32 = 0u; dy < 2u; dy = dy + 1u) {
        let y3: u32 = base_y + dy;
        if (y3 >= grid3_h) { continue; }
        for (var dx: u32 = 0u; dx < 2u; dx = dx + 1u) {
            let x3: u32 = base_x + dx;
            if (x3 >= grid3_w) { continue; }
            total_cnt = total_cnt + 1u;
            if (compute_combined_active(x3, y3, grid3_w, grid3_h)) {
                active_cnt = active_cnt + 1u;
            }
        }
    }

    let packed: u32 = (active_cnt << 16u) | (total_cnt & 0xFFFFu);
    let out_idx: u32 = gid.y * out_w + gid.x;
    if (out_idx < arrayLength(&out_mask)) {
        out_mask[out_idx] = packed;
    }
}
