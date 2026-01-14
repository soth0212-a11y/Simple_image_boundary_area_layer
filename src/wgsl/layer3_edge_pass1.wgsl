// L3Edge Pass1: continuity-based anchor bbox/score/flags (window=16, stride=4)

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>; // [height, width]
@group(0) @binding(1) var<storage, read> pooled_mask: array<u32>; // L2 output (bit0/bit1)
@group(0) @binding(2) var<storage, read_write> anchor_bbox: array<u32>;  // 2*u32 per anchor
@group(0) @binding(3) var<storage, read_write> anchor_score: array<u32>; // score per anchor
@group(0) @binding(4) var<storage, read_write> anchor_meta: array<u32>;  // valid/flags per anchor
@group(0) @binding(5) var<storage, read_write> anchor_act: array<u32>;   // act per anchor

fn expand(v: u32) -> u32 {
    return (v << 1u | v | (v >> 1u)) & 0xFFFFu;
}

fn min_bit_index(mask: u32) -> u32 {
    var idx: u32 = 16u;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        if (((mask >> i) & 1u) != 0u) {
            idx = i;
            break;
        }
    }
    return idx;
}

fn max_bit_index(mask: u32) -> u32 {
    var idx: u32 = 0u;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        if (((mask >> i) & 1u) != 0u) {
            idx = i;
        }
    }
    return idx;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&input_img_info) < 2u) { return; }
    let height: u32 = input_img_info[0];
    let width: u32 = input_img_info[1];
    if (width == 0u || height == 0u) { return; }

    let l1_w: u32 = (width + 1u) / 2u;
    let l1_h: u32 = (height + 1u) / 2u;
    var out_w: u32 = 1u;
    var out_h: u32 = 1u;
    if (l1_w >= 2u && l1_h >= 2u) {
        out_w = l1_w - 1u;
        out_h = l1_h - 1u;
    }
    if (out_w < 16u || out_h < 16u) { return; }

    let aw: u32 = ((out_w - 16u) / 4u) + 1u;
    let ah: u32 = ((out_h - 16u) / 4u) + 1u;
    if (gid.x >= aw || gid.y >= ah) { return; }

    let base_x: u32 = gid.x * 4u;
    let base_y: u32 = gid.y * 4u;

    var row_mask: array<u32, 16>;
    var col_mask: array<u32, 16>;
    var act: u32 = 0u;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        row_mask[i] = 0u;
        col_mask[i] = 0u;
    }

    for (var y: u32 = 0u; y < 16u; y = y + 1u) {
        var row: u32 = 0u;
        let gy: u32 = base_y + y;
        if (gy >= out_h) { continue; }
        for (var x: u32 = 0u; x < 16u; x = x + 1u) {
            let gx: u32 = base_x + x;
            if (gx >= out_w) { continue; }
            let idx: u32 = gy * out_w + gx;
            if (idx >= arrayLength(&pooled_mask)) { continue; }
            let m: u32 = pooled_mask[idx];
            let is_active: bool = ((m >> 1u) & 1u) != 0u;
            if (is_active) {
                row = row | (1u << x);
                col_mask[x] = col_mask[x] | (1u << y);
            }
        }
        row_mask[y] = row;
        act = act + countOneBits(row);
    }

    let touch_n: bool = row_mask[0] != 0u;
    let touch_s: bool = row_mask[15] != 0u;
    let touch_w: bool = col_mask[0] != 0u;
    let touch_e: bool = col_mask[15] != 0u;

    // Vertical scan
    var v_start_found: bool = false;
    var v_start: u32 = 0u;
    var v_end: u32 = 0u;
    var v_min_x: u32 = 16u;
    var v_max_x: u32 = 0u;
    var v_score: u32 = 0u;
    var v_frontier: u32 = 0u;
    var v_ended_by_boundary: bool = false;

    for (var y: u32 = 0u; y < 16u; y = y + 1u) {
        let pairs: u32 = row_mask[y] & (row_mask[y] >> 1u);
        if (pairs != 0u) {
            v_start_found = true;
            v_start = y;
            v_end = y;
            v_frontier = pairs;
            v_score = countOneBits(v_frontier);
            v_min_x = min_bit_index(v_frontier);
            v_max_x = max_bit_index(v_frontier);
            break;
        }
    }

    if (v_start_found) {
        var y: u32 = v_start;
        loop {
            if (y >= 15u) {
                v_ended_by_boundary = true;
                break;
            }
            let next: u32 = expand(v_frontier) & row_mask[y + 1u];
            if (next == 0u) {
                break;
            }
            y = y + 1u;
            v_frontier = next;
            v_end = y;
            v_score = v_score + countOneBits(v_frontier);
            v_min_x = min(v_min_x, min_bit_index(v_frontier));
            v_max_x = max(v_max_x, max_bit_index(v_frontier));
        }
    }

    let v_valid: bool = v_start_found && (v_end > v_start);
    let v_end_by_boundary: bool = v_valid && v_ended_by_boundary;

    // Horizontal scan
    var h_start_found: bool = false;
    var h_start: u32 = 0u;
    var h_end: u32 = 0u;
    var h_min_y: u32 = 16u;
    var h_max_y: u32 = 0u;
    var h_score: u32 = 0u;
    var h_frontier: u32 = 0u;
    var h_ended_by_boundary: bool = false;

    for (var x: u32 = 0u; x < 16u; x = x + 1u) {
        let pairs: u32 = col_mask[x] & (col_mask[x] >> 1u);
        if (pairs != 0u) {
            h_start_found = true;
            h_start = x;
            h_end = x;
            h_frontier = pairs;
            h_score = countOneBits(h_frontier);
            h_min_y = min_bit_index(h_frontier);
            h_max_y = max_bit_index(h_frontier);
            break;
        }
    }

    if (h_start_found) {
        var x: u32 = h_start;
        loop {
            if (x >= 15u) {
                h_ended_by_boundary = true;
                break;
            }
            let next: u32 = expand(h_frontier) & col_mask[x + 1u];
            if (next == 0u) {
                break;
            }
            x = x + 1u;
            h_frontier = next;
            h_end = x;
            h_score = h_score + countOneBits(h_frontier);
            h_min_y = min(h_min_y, min_bit_index(h_frontier));
            h_max_y = max(h_max_y, max_bit_index(h_frontier));
        }
    }

    let h_valid: bool = h_start_found && (h_end > h_start);
    let h_end_by_boundary: bool = h_valid && h_ended_by_boundary;

    var bbox_x0: u32 = 0u;
    var bbox_y0: u32 = 0u;
    var bbox_x1: u32 = 0u;
    var bbox_y1: u32 = 0u;
    var valid: bool = false;
    var ended_by_boundary: bool = false;

    if (v_valid && h_valid) {
        let v_x0: u32 = base_x + v_min_x;
        let v_x1: u32 = base_x + v_max_x + 1u;
        let v_y0: u32 = base_y + v_start;
        let v_y1: u32 = base_y + v_end + 1u;
        let h_x0: u32 = base_x + h_start;
        let h_x1: u32 = base_x + h_end + 1u;
        let h_y0: u32 = base_y + h_min_y;
        let h_y1: u32 = base_y + h_max_y + 1u;
        bbox_x0 = min(v_x0, h_x0);
        bbox_y0 = min(v_y0, h_y0);
        bbox_x1 = max(v_x1, h_x1);
        bbox_y1 = max(v_y1, h_y1);
        valid = true;
        ended_by_boundary = v_end_by_boundary || h_end_by_boundary;
    } else if (v_valid) {
        bbox_x0 = base_x + v_min_x;
        bbox_x1 = base_x + v_max_x + 1u;
        bbox_y0 = base_y + v_start;
        bbox_y1 = base_y + v_end + 1u;
        valid = true;
        ended_by_boundary = v_end_by_boundary;
    } else if (h_valid) {
        bbox_x0 = base_x + h_start;
        bbox_x1 = base_x + h_end + 1u;
        bbox_y0 = base_y + h_min_y;
        bbox_y1 = base_y + h_max_y + 1u;
        valid = true;
        ended_by_boundary = h_end_by_boundary;
    }

    let score: u32 = v_score + h_score;
    let idx: u32 = gid.y * aw + gid.x;
    let bbox_idx: u32 = idx * 2u;

    if (act < 8u) {
        valid = false;
        ended_by_boundary = false;
        bbox_x0 = 0u;
        bbox_y0 = 0u;
        bbox_x1 = 0u;
        bbox_y1 = 0u;
    }

    if (bbox_idx + 1u < arrayLength(&anchor_bbox)) {
        anchor_bbox[bbox_idx] = (bbox_x0 & 0xFFFFu) | ((bbox_y0 & 0xFFFFu) << 16u);
        anchor_bbox[bbox_idx + 1u] = (bbox_x1 & 0xFFFFu) | ((bbox_y1 & 0xFFFFu) << 16u);
    }
    if (idx < arrayLength(&anchor_score)) {
        anchor_score[idx] = select(0u, score, valid);
    }
    if (idx < arrayLength(&anchor_act)) {
        anchor_act[idx] = act;
    }
    if (idx < arrayLength(&anchor_meta)) {
        var flags: u32 = 0u;
        if (valid) { flags = flags | 1u; }
        if (ended_by_boundary) { flags = flags | 2u; }
        if (v_valid) { flags = flags | 4u; }
        if (h_valid) { flags = flags | 8u; }
        if (touch_n) { flags = flags | 16u; }
        if (touch_e) { flags = flags | 32u; }
        if (touch_s) { flags = flags | 64u; }
        if (touch_w) { flags = flags | 128u; }
        anchor_meta[idx] = flags;
    }
}
