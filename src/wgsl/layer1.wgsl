// L1 compute shader (1:1 grid, stride=1)
// - gid.xy -> origin_cell (x3,y3) = (sx, sy)
// - idx = y3*grid3_w + x3
// - out_mask 비트: bit0 INACTIVE, bit1 ACTIVE, bit3..6 LOCK
// - 모든 연산은 정수(u32/i32)만 사용

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>; // [height, width]
@group(0) @binding(1) var<storage, read> layer0_masks: array<u32>;   // 1:1 u32 packed (R/G/B 8방향)
@group(0) @binding(2) var<storage, read_write> out_mask: array<u32>; // packed mask

const DIR_ACTIVE_TH: u32 = 2u;

const BIT_N: u32 = 0u;
const BIT_NE: u32 = 1u;
const BIT_E: u32 = 2u;
const BIT_SE: u32 = 3u;
const BIT_S: u32 = 4u;
const BIT_SW: u32 = 5u;
const BIT_W: u32 = 6u;
const BIT_NW: u32 = 7u;

const MASK_INACTIVE: u32 = 1u << 0u;
const MASK_ACTIVE: u32 = 1u << 1u;
const LOCK_N: u32 = 1u << 3u;
const LOCK_E: u32 = 1u << 4u;
const LOCK_S: u32 = 1u << 5u;
const LOCK_W: u32 = 1u << 6u;

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

    let dispatch_w: u32 = grid3_w;
    let dispatch_h: u32 = grid3_h;
    if (gid.x >= dispatch_w || gid.y >= dispatch_h) {
        return;
    }
    // stride=1: origin_cell = (sx, sy)
    let x3: u32 = gid.x;
    let y3: u32 = gid.y;
    if (x3 >= grid3_w || y3 >= grid3_h) {
        return;
    }

    // thread 1:1 매핑
    let idx: u32 = y3 * grid3_w + x3;
    if (idx >= arrayLength(&out_mask)) { return; }

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

    var mask: u32 = 0u;
    mask = mask | select(MASK_INACTIVE, MASK_ACTIVE, combined_active);

    let cnt_n_any: u32 = max(max(cnt_n_r, cnt_n_g), cnt_n_b);
    let cnt_ne_any: u32 = max(max(cnt_ne_r, cnt_ne_g), cnt_ne_b);
    let cnt_e_any: u32 = max(max(cnt_e_r, cnt_e_g), cnt_e_b);
    let cnt_se_any: u32 = max(max(cnt_se_r, cnt_se_g), cnt_se_b);
    let cnt_s_any: u32 = max(max(cnt_s_r, cnt_s_g), cnt_s_b);
    let cnt_sw_any: u32 = max(max(cnt_sw_r, cnt_sw_g), cnt_sw_b);
    let cnt_w_any: u32 = max(max(cnt_w_r, cnt_w_g), cnt_w_b);
    let cnt_nw_any: u32 = max(max(cnt_nw_r, cnt_nw_g), cnt_nw_b);
    let all_dirs_active: bool =
        (cnt_n_any >= DIR_ACTIVE_TH) &&
        (cnt_ne_any >= DIR_ACTIVE_TH) &&
        (cnt_e_any >= DIR_ACTIVE_TH) &&
        (cnt_se_any >= DIR_ACTIVE_TH) &&
        (cnt_s_any >= DIR_ACTIVE_TH) &&
        (cnt_sw_any >= DIR_ACTIVE_TH) &&
        (cnt_w_any >= DIR_ACTIVE_TH) &&
        (cnt_nw_any >= DIR_ACTIVE_TH);
    mask = mask | select(0u, LOCK_N, cnt_n_any >= DIR_ACTIVE_TH);
    mask = mask | select(0u, LOCK_E, cnt_e_any >= DIR_ACTIVE_TH);
    mask = mask | select(0u, LOCK_S, cnt_s_any >= DIR_ACTIVE_TH);
    mask = mask | select(0u, LOCK_W, cnt_w_any >= DIR_ACTIVE_TH);

    if (all_dirs_active) {
        out_mask[idx] = 0u;
    } else {
        out_mask[idx] = mask;
    }
}
