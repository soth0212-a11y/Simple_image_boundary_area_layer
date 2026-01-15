struct L4Params {
    ring_pct: u32,
    iou_tenths: u32,
    gap_max: u32,
    ov_pct: u32,
}

const BIN: u32 = 64u;
const BIN_CAP: u32 = 64u;
const INVALID: u32 = 0xFFFFFFFFu;

fn compute_l2_dims(height: u32, width: u32) -> vec2<u32> {
    let l1_w: u32 = (width + 1u) / 2u;
    let l1_h: u32 = (height + 1u) / 2u;
    var out_w: u32 = 1u;
    var out_h: u32 = 1u;
    if (l1_w >= 2u && l1_h >= 2u) {
        out_w = l1_w - 1u;
        out_h = l1_h - 1u;
    }
    return vec2<u32>(out_w, out_h);
}

fn is_active(idx: u32) -> bool {
    if (idx >= arrayLength(&l2_mask)) {
        return false;
    }
    return (l2_mask[idx] & (1u << 1u)) != 0u;
}

struct InputInfo {
    height: u32,
    width: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> input_img_info: InputInfo;
@group(0) @binding(1) var<storage, read> l2_mask: array<u32>;
@group(0) @binding(2) var<storage, read> l3_boxes: array<u32>;
@group(0) @binding(3) var<storage, read> l3_valid: array<u32>;
@group(0) @binding(4) var<storage, read> params: L4Params;
@group(0) @binding(5) var<storage, read_write> bin_counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> bin_items: array<u32>;
@group(0) @binding(7) var<storage, read_write> expanded_boxes: array<u32>;
@group(0) @binding(8) var<storage, read_write> expanded_flags: array<u32>;

fn decode_bbox(idx: u32, l2_w: u32, l2_h: u32) -> vec4<u32> {
    let bbox_idx: u32 = idx * 2u;
    if (bbox_idx + 1u >= arrayLength(&l3_boxes)) {
        return vec4<u32>(0u, 0u, 0u, 0u);
    }
    let b0: u32 = l3_boxes[bbox_idx];
    let b1: u32 = l3_boxes[bbox_idx + 1u];
    let x0: u32 = b0 & 0xFFFFu;
    let y0: u32 = b0 >> 16u;
    let x1_ex: u32 = b1 & 0xFFFFu;
    let y1_ex: u32 = b1 >> 16u;
    if (x1_ex == 0u || y1_ex == 0u) {
        return vec4<u32>(0u, 0u, 0u, 0u);
    }
    let x1: u32 = min(x1_ex - 1u, l2_w - 1u);
    let y1: u32 = min(y1_ex - 1u, l2_h - 1u);
    if (x1 < x0 || y1 < y0) {
        return vec4<u32>(0u, 0u, 0u, 0u);
    }
    return vec4<u32>(x0, y0, x1, y1);
}

fn iou_inter(a: vec4<u32>, b: vec4<u32>) -> u32 {
    let ix0: u32 = max(a.x, b.x);
    let iy0: u32 = max(a.y, b.y);
    let ix1: u32 = min(a.z, b.z);
    let iy1: u32 = min(a.w, b.w);
    if (ix1 < ix0 || iy1 < iy0) {
        return 0u;
    }
    return (ix1 - ix0 + 1u) * (iy1 - iy0 + 1u);
}

fn area_inclusive(a: vec4<u32>) -> u32 {
    return (a.z - a.x + 1u) * (a.w - a.y + 1u);
}

fn adj_score_right(a: vec4<u32>, b: vec4<u32>, gap_max: u32, ov_pct: u32) -> u32 {
    if (b.x <= a.z) { return 0u; }
    let gap: u32 = b.x - a.z - 1u;
    if (gap > gap_max) { return 0u; }
    let ov0: u32 = max(a.y, b.y);
    let ov1: u32 = min(a.w, b.w);
    if (ov1 < ov0) { return 0u; }
    let ov: u32 = ov1 - ov0 + 1u;
    let ha: u32 = a.w - a.y + 1u;
    let hb: u32 = b.w - b.y + 1u;
    let denom: u32 = min(ha, hb);
    if (ov * 100u < denom * ov_pct) { return 0u; }
    let base: u32 = ov * 1024u;
    return select(0u, base - gap, base > gap);
}

fn adj_score_left(a: vec4<u32>, b: vec4<u32>, gap_max: u32, ov_pct: u32) -> u32 {
    if (b.z >= a.x) { return 0u; }
    let gap: u32 = a.x - b.z - 1u;
    if (gap > gap_max) { return 0u; }
    let ov0: u32 = max(a.y, b.y);
    let ov1: u32 = min(a.w, b.w);
    if (ov1 < ov0) { return 0u; }
    let ov: u32 = ov1 - ov0 + 1u;
    let ha: u32 = a.w - a.y + 1u;
    let hb: u32 = b.w - b.y + 1u;
    let denom: u32 = min(ha, hb);
    if (ov * 100u < denom * ov_pct) { return 0u; }
    let base: u32 = ov * 1024u;
    return select(0u, base - gap, base > gap);
}

fn adj_score_bottom(a: vec4<u32>, b: vec4<u32>, gap_max: u32, ov_pct: u32) -> u32 {
    if (b.y <= a.w) { return 0u; }
    let gap: u32 = b.y - a.w - 1u;
    if (gap > gap_max) { return 0u; }
    let ov0: u32 = max(a.x, b.x);
    let ov1: u32 = min(a.z, b.z);
    if (ov1 < ov0) { return 0u; }
    let ov: u32 = ov1 - ov0 + 1u;
    let wa: u32 = a.z - a.x + 1u;
    let wb: u32 = b.z - b.x + 1u;
    let denom: u32 = min(wa, wb);
    if (ov * 100u < denom * ov_pct) { return 0u; }
    let base: u32 = ov * 1024u;
    return select(0u, base - gap, base > gap);
}

fn adj_score_top(a: vec4<u32>, b: vec4<u32>, gap_max: u32, ov_pct: u32) -> u32 {
    if (b.w >= a.y) { return 0u; }
    let gap: u32 = a.y - b.w - 1u;
    if (gap > gap_max) { return 0u; }
    let ov0: u32 = max(a.x, b.x);
    let ov1: u32 = min(a.z, b.z);
    if (ov1 < ov0) { return 0u; }
    let ov: u32 = ov1 - ov0 + 1u;
    let wa: u32 = a.z - a.x + 1u;
    let wb: u32 = b.z - b.x + 1u;
    let denom: u32 = min(wa, wb);
    if (ov * 100u < denom * ov_pct) { return 0u; }
    let base: u32 = ov * 1024u;
    return select(0u, base - gap, base > gap);
}

@compute @workgroup_size(256, 1, 1)
fn clear_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    if (idx >= arrayLength(&bin_counts)) { return; }
    atomicStore(&bin_counts[idx], 0u);
}

@compute @workgroup_size(256, 1, 1)
fn insert_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let height: u32 = input_img_info.height;
    let width: u32 = input_img_info.width;
    if (width == 0u || height == 0u) { return; }
    let l2_dims = compute_l2_dims(height, width);
    let l2_w: u32 = l2_dims.x;
    let l2_h: u32 = l2_dims.y;
    if (l2_w == 0u || l2_h == 0u) { return; }

    let idx: u32 = gid.x;
    if (idx >= arrayLength(&l3_valid)) { return; }
    if ((l3_valid[idx] & 1u) == 0u) { return; }

    let b = decode_bbox(idx, l2_w, l2_h);
    if (b.z < b.x || b.w < b.y) { return; }
    let cx: u32 = (b.x + b.z) / 2u;
    let cy: u32 = (b.y + b.w) / 2u;
    let bin_w: u32 = (l2_w + BIN - 1u) / BIN;
    let bin_h: u32 = (l2_h + BIN - 1u) / BIN;
    let bx: u32 = cx / BIN;
    let by: u32 = cy / BIN;
    if (bx >= bin_w || by >= bin_h) { return; }
    let bin_idx: u32 = by * bin_w + bx;
    if (bin_idx >= arrayLength(&bin_counts)) { return; }
    let slot: u32 = atomicAdd(&bin_counts[bin_idx], 1u);
    if (slot >= BIN_CAP) { return; }
    let base: u32 = bin_idx * BIN_CAP + slot;
    if (base >= arrayLength(&bin_items)) { return; }
    bin_items[base] = idx;
}

@compute @workgroup_size(256, 1, 1)
fn expand_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let height: u32 = input_img_info.height;
    let width: u32 = input_img_info.width;
    if (width == 0u || height == 0u) { return; }

    let l2_dims = compute_l2_dims(height, width);
    let l2_w: u32 = l2_dims.x;
    let l2_h: u32 = l2_dims.y;
    if (l2_w == 0u || l2_h == 0u) { return; }

    let idx: u32 = gid.x;
    let bbox_idx: u32 = idx * 2u;
    if (bbox_idx + 1u >= arrayLength(&l3_boxes) || idx >= arrayLength(&l3_valid)) { return; }
    if (idx >= arrayLength(&expanded_flags)) { return; }

    let v: u32 = l3_valid[idx];
    if ((v & 1u) == 0u) {
        if (bbox_idx + 1u < arrayLength(&expanded_boxes)) {
            expanded_boxes[bbox_idx] = 0u;
            expanded_boxes[bbox_idx + 1u] = 0u;
        }
        expanded_flags[idx] = 0u;
        return;
    }

    let b = decode_bbox(idx, l2_w, l2_h);
    var x0: u32 = b.x;
    var y0: u32 = b.y;
    var x1: u32 = b.z;
    var y1: u32 = b.w;
    if (x1 < x0 || y1 < y0) {
        expanded_flags[idx] = 0u;
        return;
    }

    var right_cnt: u32 = 0u;
    var left_cnt: u32 = 0u;
    var bottom_cnt: u32 = 0u;
    var top_cnt: u32 = 0u;
    var right_len: u32 = 0u;
    var left_len: u32 = 0u;
    var bottom_len: u32 = 0u;
    var top_len: u32 = 0u;

    if (x1 + 1u < l2_w) {
        right_len = y1 - y0 + 1u;
        let x: u32 = x1 + 1u;
        for (var y: u32 = y0; y <= y1; y = y + 1u) {
            let idx2: u32 = y * l2_w + x;
            if (is_active(idx2)) { right_cnt = right_cnt + 1u; }
        }
    }
    if (x0 > 0u) {
        left_len = y1 - y0 + 1u;
        let x: u32 = x0 - 1u;
        for (var y: u32 = y0; y <= y1; y = y + 1u) {
            let idx2: u32 = y * l2_w + x;
            if (is_active(idx2)) { left_cnt = left_cnt + 1u; }
        }
    }
    if (y1 + 1u < l2_h) {
        bottom_len = x1 - x0 + 1u;
        let y: u32 = y1 + 1u;
        for (var x: u32 = x0; x <= x1; x = x + 1u) {
            let idx2: u32 = y * l2_w + x;
            if (is_active(idx2)) { bottom_cnt = bottom_cnt + 1u; }
        }
    }
    if (y0 > 0u) {
        top_len = x1 - x0 + 1u;
        let y: u32 = y0 - 1u;
        for (var x: u32 = x0; x <= x1; x = x + 1u) {
            let idx2: u32 = y * l2_w + x;
            if (is_active(idx2)) { top_cnt = top_cnt + 1u; }
        }
    }

    let ring_pct: u32 = params.ring_pct;
    let right_pass: bool = right_len > 0u && right_cnt * 100u >= right_len * ring_pct;
    let left_pass: bool = left_len > 0u && left_cnt * 100u >= left_len * ring_pct;
    let bottom_pass: bool = bottom_len > 0u && bottom_cnt * 100u >= bottom_len * ring_pct;
    let top_pass: bool = top_len > 0u && top_cnt * 100u >= top_len * ring_pct;

    var cand: array<vec4<u32>, 4>;
    let x1r: u32 = min(x1 + 1u, l2_w - 1u);
    let y1b: u32 = min(y1 + 1u, l2_h - 1u);
    var x0l: u32 = x0;
    var y0t: u32 = y0;
    if (x0 > 0u) { x0l = x0 - 1u; }
    if (y0 > 0u) { y0t = y0 - 1u; }
    cand[0] = vec4<u32>(x0, y0, x1r, y1);
    cand[1] = vec4<u32>(x0l, y0, x1, y1);
    cand[2] = vec4<u32>(x0, y0, x1, y1b);
    cand[3] = vec4<u32>(x0, y0t, x1, y1);

    var dir_ok: array<u32, 4>;
    var dir_score: array<u32, 4>;
    var dir_iou: array<u32, 4>;
    var dir_neighbor: array<u32, 4>;
    for (var d: u32 = 0u; d < 4u; d = d + 1u) {
        dir_ok[d] = 0u;
        dir_score[d] = 0u;
        dir_iou[d] = 0u;
        dir_neighbor[d] = INVALID;
    }

    let bin_w: u32 = (l2_w + BIN - 1u) / BIN;
    let bin_h: u32 = (l2_h + BIN - 1u) / BIN;
    let cx: u32 = (x0 + x1) / 2u;
    let cy: u32 = (y0 + y1) / 2u;
    let bx: i32 = i32(cx / BIN);
    let by: i32 = i32(cy / BIN);

    let dir_pass = array<u32, 4>(
        select(0u, 1u, right_pass),
        select(0u, 1u, left_pass),
        select(0u, 1u, bottom_pass),
        select(0u, 1u, top_pass)
    );
    let dir_valid = array<u32, 4>(
        select(0u, 1u, x1 + 1u < l2_w),
        select(0u, 1u, x0 > 0u),
        select(0u, 1u, y1 + 1u < l2_h),
        select(0u, 1u, y0 > 0u)
    );

    for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
        let gy: i32 = by + oy;
        if (gy < 0 || gy >= i32(bin_h)) { continue; }
        for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
            let gx: i32 = bx + ox;
            if (gx < 0 || gx >= i32(bin_w)) { continue; }
            let bin_idx: u32 = u32(gy) * bin_w + u32(gx);
            if (bin_idx >= arrayLength(&bin_counts)) { continue; }
            let count: u32 = min(atomicLoad(&bin_counts[bin_idx]), BIN_CAP);
            let base: u32 = bin_idx * BIN_CAP;
            for (var k: u32 = 0u; k < count; k = k + 1u) {
                let item_idx: u32 = base + k;
                if (item_idx >= arrayLength(&bin_items)) { break; }
                let nb_idx: u32 = bin_items[item_idx];
                if (nb_idx == INVALID || nb_idx == idx) { continue; }
                if (nb_idx >= arrayLength(&l3_valid)) { continue; }
                if ((l3_valid[nb_idx] & 1u) == 0u) { continue; }
                let nb = decode_bbox(nb_idx, l2_w, l2_h);
                if (nb.z < nb.x || nb.w < nb.y) { continue; }

                for (var d: u32 = 0u; d < 4u; d = d + 1u) {
                    if (dir_valid[d] == 0u || dir_pass[d] == 0u) { continue; }
                    let c = cand[d];
                    let inter: u32 = iou_inter(c, nb);
                    let area_c: u32 = area_inclusive(c);
                    let area_nb: u32 = area_inclusive(nb);
                    let uni: u32 = area_c + area_nb - inter;
                    var score: u32 = 0u;
                    var passed: bool = false;
                    var by_iou: bool = false;
                    if (uni > 0u && inter * 10u >= uni * params.iou_tenths) {
                        score = inter;
                        passed = true;
                        by_iou = true;
                    } else {
                        if (d == 0u) { score = adj_score_right(c, nb, params.gap_max, params.ov_pct); }
                        if (d == 1u) { score = adj_score_left(c, nb, params.gap_max, params.ov_pct); }
                        if (d == 2u) { score = adj_score_bottom(c, nb, params.gap_max, params.ov_pct); }
                        if (d == 3u) { score = adj_score_top(c, nb, params.gap_max, params.ov_pct); }
                        if (score > 0u) {
                            passed = true;
                            by_iou = false;
                        }
                    }
                    if (!passed) { continue; }
                    let best_score: u32 = dir_score[d];
                    let best_nb: u32 = dir_neighbor[d];
                    if (score > best_score || (score == best_score && nb_idx < best_nb)) {
                        dir_ok[d] = 1u;
                        dir_score[d] = score;
                        dir_iou[d] = select(0u, 1u, by_iou);
                        dir_neighbor[d] = nb_idx;
                    }
                }
            }
        }
    }

    var best1_dir: i32 = -1;
    var best2_dir: i32 = -1;
    var best1_score: u32 = 0u;
    var best2_score: u32 = 0u;
    let order = array<u32, 4>(0u, 1u, 2u, 3u);
    for (var oi: u32 = 0u; oi < 4u; oi = oi + 1u) {
        let d: u32 = order[oi];
        if (dir_ok[d] == 0u) { continue; }
        let score: u32 = dir_score[d];
        if (best1_dir < 0 || score > best1_score) {
            best2_dir = best1_dir;
            best2_score = best1_score;
            best1_dir = i32(d);
            best1_score = score;
        } else if (score == best1_score) {
            if (best2_dir < 0 || score > best2_score) {
                best2_dir = i32(d);
                best2_score = score;
            }
        } else if (best2_dir < 0 || score > best2_score) {
            best2_dir = i32(d);
            best2_score = score;
        }
    }

    if (best1_dir >= 0) {
        let d: u32 = u32(best1_dir);
        if (d == 0u && x1 + 1u < l2_w) { x1 = x1 + 1u; }
        if (d == 1u && x0 > 0u) { x0 = x0 - 1u; }
        if (d == 2u && y1 + 1u < l2_h) { y1 = y1 + 1u; }
        if (d == 3u && y0 > 0u) { y0 = y0 - 1u; }
    }
    if (best2_dir >= 0) {
        let d: u32 = u32(best2_dir);
        if (d == 0u && x1 + 1u < l2_w) { x1 = x1 + 1u; }
        if (d == 1u && x0 > 0u) { x0 = x0 - 1u; }
        if (d == 2u && y1 + 1u < l2_h) { y1 = y1 + 1u; }
        if (d == 3u && y0 > 0u) { y0 = y0 - 1u; }
    }

    if (bbox_idx + 1u < arrayLength(&expanded_boxes)) {
        expanded_boxes[bbox_idx] = (x0 & 0xFFFFu) | ((y0 & 0xFFFFu) << 16u);
        expanded_boxes[bbox_idx + 1u] = (x1 & 0xFFFFu) | ((y1 & 0xFFFFu) << 16u);
    }
    var flags: u32 = 0u;
    var primary: u32 = 3u;
    var secondary: u32 = 3u;
    if (best1_dir >= 0) { primary = u32(best1_dir); }
    if (best2_dir >= 0) { secondary = u32(best2_dir); }
    if (best1_dir >= 0) { flags = flags | 1u; }
    flags = flags | ((primary & 3u) << 1u);
    flags = flags | ((secondary & 3u) << 3u);
    if (best1_dir >= 0 && dir_iou[u32(best1_dir)] != 0u) { flags = flags | (1u << 5u); }
    if (best2_dir >= 0 && dir_iou[u32(best2_dir)] != 0u) { flags = flags | (1u << 6u); }
    flags = flags | (1u << 7u);
    expanded_flags[idx] = flags;
}
