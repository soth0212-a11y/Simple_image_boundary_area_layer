// L3Edge Pass2a: 4x4 block top-10 by active density

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>; // [height, width]
@group(0) @binding(1) var<storage, read> anchor_bbox: array<u32>;
@group(0) @binding(2) var<storage, read> anchor_score: array<u32>;
@group(0) @binding(3) var<storage, read> anchor_meta: array<u32>;
@group(0) @binding(4) var<storage, read> anchor_act: array<u32>;
@group(0) @binding(5) var<storage, read_write> stage1_boxes: array<u32>;  // 10 slots * 2
@group(0) @binding(6) var<storage, read_write> stage1_scores: array<u32>; // 10 slots (density)
@group(0) @binding(7) var<storage, read_write> stage1_valid: array<u32>;  // 10 slots

struct Cand {
    valid: bool,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    edge_score: u32,
    area: u32,
    id: u32,
    flags: u32,
    act: u32,
    density: u32,
}

const FLAG_VALID: u32 = 1u;

fn bbox_area(x0: u32, y0: u32, x1: u32, y1: u32) -> u32 {
    if (x1 <= x0 || y1 <= y0) { return 0u; }
    return (x1 - x0) * (y1 - y0);
}

fn read_anchor(ax: u32, ay: u32, aw: u32, ah: u32, mean_act: u32) -> Cand {
    if (ax >= aw || ay >= ah) {
        return Cand(false, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    let idx: u32 = ay * aw + ax;
    if (idx >= arrayLength(&anchor_meta)) {
        return Cand(false, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    let flags: u32 = anchor_meta[idx];
    if ((flags & FLAG_VALID) == 0u) {
        return Cand(false, 0u, 0u, 0u, 0u, 0u, 0u, idx, flags, 0u, 0u);
    }
    var act: u32 = 0u;
    if (idx < arrayLength(&anchor_act)) {
        act = anchor_act[idx];
        if (act * 100u < mean_act * 60u) {
            return Cand(false, 0u, 0u, 0u, 0u, 0u, 0u, idx, flags, act, 0u);
        }
    }
    let bbox_idx: u32 = idx * 2u;
    if (bbox_idx + 1u >= arrayLength(&anchor_bbox)) {
        return Cand(false, 0u, 0u, 0u, 0u, 0u, 0u, idx, flags, act, 0u);
    }
    let b0: u32 = anchor_bbox[bbox_idx];
    let b1: u32 = anchor_bbox[bbox_idx + 1u];
    let x0: u32 = b0 & 0xFFFFu;
    let y0: u32 = b0 >> 16u;
    let x1: u32 = b1 & 0xFFFFu;
    let y1: u32 = b1 >> 16u;
    var edge_score: u32 = 0u;
    if (idx < arrayLength(&anchor_score)) {
        edge_score = anchor_score[idx];
    }
    let area: u32 = bbox_area(x0, y0, x1, y1);
    if (area == 0u) {
        return Cand(false, 0u, 0u, 0u, 0u, edge_score, 0u, idx, flags, act, 0u);
    }
    let density: u32 = (act * 1000u) / area;
    return Cand(true, x0, y0, x1, y1, edge_score, area, idx, flags, act, density);
}

fn better_density(a: Cand, b: Cand) -> bool {
    if (a.density != b.density) { return a.density > b.density; }
    if (a.edge_score != b.edge_score) { return a.edge_score > b.edge_score; }
    if (a.area != b.area) { return a.area > b.area; }
    return a.id < b.id;
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
    if (gid.x >= bw || gid.y >= bh) { return; }

    let base_ax: u32 = gid.x * 4u;
    let base_ay: u32 = gid.y * 4u;

    var sum_act: u32 = 0u;
    var cnt_valid: u32 = 0u;
    for (var dy: u32 = 0u; dy < 4u; dy = dy + 1u) {
        let ay: u32 = base_ay + dy;
        if (ay >= ah) { continue; }
        for (var dx: u32 = 0u; dx < 4u; dx = dx + 1u) {
            let ax: u32 = base_ax + dx;
            if (ax >= aw) { continue; }
            let idx: u32 = ay * aw + ax;
            if (idx >= arrayLength(&anchor_meta)) { continue; }
            let flags: u32 = anchor_meta[idx];
            if ((flags & FLAG_VALID) == 0u) { continue; }
            if (idx < arrayLength(&anchor_act)) {
                sum_act = sum_act + anchor_act[idx];
                cnt_valid = cnt_valid + 1u;
            }
        }
    }
    if (cnt_valid == 0u) { return; }
    let mean_act: u32 = sum_act / cnt_valid;

    let block_idx: u32 = gid.y * bw + gid.x;
    var top: array<Cand, 10>;
    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
        top[i] = Cand(false, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }

    for (var dy: u32 = 0u; dy < 4u; dy = dy + 1u) {
        let ay: u32 = base_ay + dy;
        for (var dx: u32 = 0u; dx < 4u; dx = dx + 1u) {
            let ax: u32 = base_ax + dx;
            let c: Cand = read_anchor(ax, ay, aw, ah, mean_act);
            if (!c.valid) { continue; }
            for (var i: u32 = 0u; i < 10u; i = i + 1u) {
                if (!top[i].valid || better_density(c, top[i])) {
                    var j: i32 = 9;
                    loop {
                        if (j <= i32(i)) { break; }
                        let uj: u32 = u32(j);
                        let ujm1: u32 = u32(j - 1);
                        top[uj] = top[ujm1];
                        j = j - 1;
                    }
                    top[i] = c;
                    break;
                }
            }
        }
    }

    let base_out: u32 = block_idx * 10u;
    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
        let out_idx: u32 = base_out + i;
        let bbox_idx: u32 = out_idx * 2u;
        let c: Cand = top[i];
        let valid: bool = c.valid;
        if (bbox_idx + 1u < arrayLength(&stage1_boxes)) {
            stage1_boxes[bbox_idx] = select(0u, (c.x0 & 0xFFFFu) | ((c.y0 & 0xFFFFu) << 16u), valid);
            stage1_boxes[bbox_idx + 1u] = select(0u, (c.x1 & 0xFFFFu) | ((c.y1 & 0xFFFFu) << 16u), valid);
        }
        if (out_idx < arrayLength(&stage1_scores)) {
            stage1_scores[out_idx] = select(0u, c.density, valid);
        }
        if (out_idx < arrayLength(&stage1_valid)) {
            stage1_valid[out_idx] = select(0u, c.flags | 1u, valid);
        }
    }
}
