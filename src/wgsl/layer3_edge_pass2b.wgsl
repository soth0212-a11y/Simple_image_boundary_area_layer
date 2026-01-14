// L3Edge Pass2b: merge 2x2 blocks (8x8 anchors) output 40 slots

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>; // [height, width]
@group(0) @binding(1) var<storage, read> stage1_boxes: array<u32>;  // 10 slots * 2
@group(0) @binding(2) var<storage, read> stage1_scores: array<u32>; // 10 slots
@group(0) @binding(3) var<storage, read> stage1_valid: array<u32>;  // 10 slots (flags|1)
@group(0) @binding(4) var<storage, read_write> block_boxes: array<u32>;  // 40 slots * 2
@group(0) @binding(5) var<storage, read_write> block_scores: array<u32>; // 40 slots
@group(0) @binding(6) var<storage, read_write> block_valid: array<u32>;  // 40 slots

struct Cand {
    valid: bool,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    score: u32,
    area: u32,
    id: u32,
    flags: u32,
}

const FLAG_TOUCH_N: u32 = 16u;
const FLAG_TOUCH_E: u32 = 32u;
const FLAG_TOUCH_S: u32 = 64u;
const FLAG_TOUCH_W: u32 = 128u;

fn bbox_area(x0: u32, y0: u32, x1: u32, y1: u32) -> u32 {
    if (x1 <= x0 || y1 <= y0) { return 0u; }
    return (x1 - x0) * (y1 - y0);
}

fn iou_ge_t2(a: Cand, b: Cand) -> bool {
    let inter_x0: u32 = max(a.x0, b.x0);
    let inter_y0: u32 = max(a.y0, b.y0);
    let inter_x1: u32 = min(a.x1, b.x1);
    let inter_y1: u32 = min(a.y1, b.y1);
    if (inter_x1 <= inter_x0 || inter_y1 <= inter_y0) { return false; }
    let inter_area: u32 = (inter_x1 - inter_x0) * (inter_y1 - inter_y0);
    let union_area: u32 = a.area + b.area - inter_area;
    return inter_area * 10u >= union_area * 9u;
}

fn better(score_a: u32, area_a: u32, id_a: u32, score_b: u32, area_b: u32, id_b: u32) -> bool {
    if (score_a != score_b) { return score_a > score_b; }
    if (area_a != area_b) { return area_a > area_b; }
    return id_a < id_b;
}

fn union_boxes(a: Cand, b: Cand) -> Cand {
    var out: Cand = a;
    out.x0 = min(a.x0, b.x0);
    out.y0 = min(a.y0, b.y0);
    out.x1 = max(a.x1, b.x1);
    out.y1 = max(a.y1, b.y1);
    out.score = max(a.score, b.score);
    out.area = bbox_area(out.x0, out.y0, out.x1, out.y1);
    out.id = min(a.id, b.id);
    out.flags = a.flags | b.flags;
    out.valid = true;
    return out;
}

fn allow_merge(a: Cand, b: Cand) -> bool {
    let ax: u32 = (a.x0 + a.x1) / 2u;
    let ay: u32 = (a.y0 + a.y1) / 2u;
    let bx: u32 = (b.x0 + b.x1) / 2u;
    let by: u32 = (b.y0 + b.y1) / 2u;
    let dx: i32 = i32(bx) - i32(ax);
    let dy: i32 = i32(by) - i32(ay);
    if (dx == 0 && dy == 0) {
        return true;
    }

    let a_n: bool = (a.flags & FLAG_TOUCH_N) != 0u;
    let a_e: bool = (a.flags & FLAG_TOUCH_E) != 0u;
    let a_s: bool = (a.flags & FLAG_TOUCH_S) != 0u;
    let a_w: bool = (a.flags & FLAG_TOUCH_W) != 0u;
    let b_n: bool = (b.flags & FLAG_TOUCH_N) != 0u;
    let b_e: bool = (b.flags & FLAG_TOUCH_E) != 0u;
    let b_s: bool = (b.flags & FLAG_TOUCH_S) != 0u;
    let b_w: bool = (b.flags & FLAG_TOUCH_W) != 0u;

    if (dx == 0) {
        if (dy > 0) { return a_s && b_n; }
        return a_n && b_s;
    }
    if (dy == 0) {
        if (dx > 0) { return a_e && b_w; }
        return a_w && b_e;
    }

    if (dx > 0 && dy < 0) { return a_n && a_e && b_s && b_w; }
    if (dx > 0 && dy > 0) { return a_s && a_e && b_n && b_w; }
    if (dx < 0 && dy < 0) { return a_n && a_w && b_s && b_e; }
    return a_s && a_w && b_n && b_e;
}

fn find_root(parent: ptr<function, array<u32, 40>>, idx: u32) -> u32 {
    var x: u32 = idx;
    loop {
        let p: u32 = (*parent)[x];
        if (p == x) { break; }
        x = p;
    }
    return x;
}

fn union_root(parent: ptr<function, array<u32, 40>>, a: u32, b: u32) {
    let ra: u32 = find_root(parent, a);
    let rb: u32 = find_root(parent, b);
    if (ra != rb) {
        (*parent)[rb] = ra;
    }
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
    let bw: u32 = (aw + 3u) / 4u;
    let bh: u32 = (ah + 3u) / 4u;
    let bw2: u32 = (bw + 1u) / 2u;
    let bh2: u32 = (bh + 1u) / 2u;
    if (gid.x >= bw2 || gid.y >= bh2) { return; }

    let base_bx: u32 = gid.x * 2u;
    let base_by: u32 = gid.y * 2u;

    var cands: array<Cand, 40>;
    for (var i: u32 = 0u; i < 40u; i = i + 1u) {
        cands[i] = Cand(false, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    var count: u32 = 0u;

    for (var by: u32 = 0u; by < 2u; by = by + 1u) {
        let gy: u32 = base_by + by;
        if (gy >= bh) { continue; }
        for (var bx: u32 = 0u; bx < 2u; bx = bx + 1u) {
            let gx: u32 = base_bx + bx;
            if (gx >= bw) { continue; }
            let block_idx: u32 = gy * bw + gx;
            let base_slot: u32 = block_idx * 10u;
            for (var s: u32 = 0u; s < 10u; s = s + 1u) {
                let slot_idx: u32 = base_slot + s;
                if (slot_idx >= arrayLength(&stage1_valid)) { continue; }
                let flags: u32 = stage1_valid[slot_idx];
                if ((flags & 1u) == 0u) { continue; }
                if (count >= 40u) { continue; }
                let bbox_idx: u32 = slot_idx * 2u;
                if (bbox_idx + 1u >= arrayLength(&stage1_boxes)) { continue; }
                let b0: u32 = stage1_boxes[bbox_idx];
                let b1: u32 = stage1_boxes[bbox_idx + 1u];
                let x0: u32 = b0 & 0xFFFFu;
                let y0: u32 = b0 >> 16u;
                let x1: u32 = b1 & 0xFFFFu;
                let y1: u32 = b1 >> 16u;
                var score: u32 = 0u;
                if (slot_idx < arrayLength(&stage1_scores)) {
                    score = stage1_scores[slot_idx];
                }
                let area: u32 = bbox_area(x0, y0, x1, y1);
                cands[count] = Cand(true, x0, y0, x1, y1, score, area, slot_idx, flags);
                count = count + 1u;
            }
        }
    }

    var parent: array<u32, 40>;
    for (var i: u32 = 0u; i < 40u; i = i + 1u) {
        parent[i] = i;
    }

    for (var i: u32 = 0u; i < 40u; i = i + 1u) {
        if (!cands[i].valid) { continue; }
        for (var j: u32 = i + 1u; j < 40u; j = j + 1u) {
            if (!cands[j].valid) { continue; }
            if (iou_ge_t2(cands[i], cands[j]) && allow_merge(cands[i], cands[j])) {
                union_root(&parent, i, j);
            }
        }
    }

    var merged: array<Cand, 40>;
    for (var i: u32 = 0u; i < 40u; i = i + 1u) {
        merged[i] = Cand(false, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    for (var i: u32 = 0u; i < 40u; i = i + 1u) {
        if (!cands[i].valid) { continue; }
        let root: u32 = find_root(&parent, i);
        if (!merged[root].valid) {
            merged[root] = cands[i];
        } else {
            merged[root] = union_boxes(merged[root], cands[i]);
        }
    }

    let block_idx: u32 = gid.y * bw2 + gid.x;
    let base_out: u32 = block_idx * 40u;
    for (var i: u32 = 0u; i < 40u; i = i + 1u) {
        let out_idx: u32 = base_out + i;
        let bbox_idx: u32 = out_idx * 2u;
        let c: Cand = merged[i];
        let valid: bool = c.valid;
        if (bbox_idx + 1u < arrayLength(&block_boxes)) {
            block_boxes[bbox_idx] = select(0u, (c.x0 & 0xFFFFu) | ((c.y0 & 0xFFFFu) << 16u), valid);
            block_boxes[bbox_idx + 1u] = select(0u, (c.x1 & 0xFFFFu) | ((c.y1 & 0xFFFFu) << 16u), valid);
        }
        if (out_idx < arrayLength(&block_scores)) {
            block_scores[out_idx] = select(0u, c.score, valid);
        }
        if (out_idx < arrayLength(&block_valid)) {
            block_valid[out_idx] = select(0u, c.flags | 1u, valid);
        }
    }
}
