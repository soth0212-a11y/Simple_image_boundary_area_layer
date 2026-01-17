struct Params {
    total_boxes: u32,
    bins_x: u32,
    bins_y: u32,
    bin_size: u32,
    color_tol: u32,
    gap_x: u32,
    gap_y: u32,
    overlap_min: u32,
}

struct OutBox {
    x0y0: u32,
    x1y1: u32,
    color565: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> in_boxes: array<OutBox>;
@group(0) @binding(1) var<storage, read> bin_offset: array<u32>;
@group(0) @binding(2) var<storage, read> bin_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> bin_items: array<u32>;
@group(0) @binding(4) var<storage, read_write> labels: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

fn clamp_bin(v: u32, maxv: u32) -> u32 {
    if (maxv == 0u) { return 0u; }
    return min(v, maxv - 1u);
}

fn absdiff(a: u32, b: u32) -> u32 {
    return select(a - b, b - a, a >= b);
}

fn color_close(a: u32, b: u32, tol: u32) -> bool {
    let ar = (a >> 11u) & 31u;
    let ag = (a >> 5u) & 63u;
    let ab = a & 31u;
    let br = (b >> 11u) & 31u;
    let bg = (b >> 5u) & 63u;
    let bb = b & 31u;
    return absdiff(ar, br) <= tol && absdiff(ag, bg) <= tol && absdiff(ab, bb) <= tol;
}

fn overlap_1d(mina: u32, maxa: u32, minb: u32, maxb: u32) -> u32 {
    let lo = max(mina, minb);
    let hi = min(maxa, maxb);
    return select(hi - lo, 0u, hi > lo);
}

fn gap_1d(mina: u32, maxa: u32, minb: u32, maxb: u32) -> u32 {
    if (mina > maxb) { return mina - maxb; }
    if (minb > maxa) { return minb - maxa; }
    return 0u;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_boxes) { return; }
    if (idx >= arrayLength(&in_boxes)) { return; }
    let b = in_boxes[idx];
    let minx = b.x0y0 & 0xFFFFu;
    let miny = b.x0y0 >> 16u;
    let maxx = b.x1y1 & 0xFFFFu;
    let maxy = b.x1y1 >> 16u;
    let cx = (minx + maxx) >> 1u;
    let cy = (miny + maxy) >> 1u;
    let bin_x = clamp_bin(cx / params.bin_size, params.bins_x);
    let bin_y = clamp_bin(cy / params.bin_size, params.bins_y);

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        let by = i32(bin_y) + dy;
        if (by < 0 || by >= i32(params.bins_y)) { continue; }
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let bx = i32(bin_x) + dx;
            if (bx < 0 || bx >= i32(params.bins_x)) { continue; }
            let bin_id = u32(by) * params.bins_x + u32(bx);
            if (bin_id >= arrayLength(&bin_offset) || bin_id >= arrayLength(&bin_count)) { continue; }
            let start = bin_offset[bin_id];
            let count = atomicLoad(&bin_count[bin_id]);
            for (var k: u32 = 0u; k < count; k = k + 1u) {
                let item_idx = start + k;
                if (item_idx >= arrayLength(&bin_items)) { break; }
                let j = bin_items[item_idx];
                if (j == idx || j >= params.total_boxes || j >= arrayLength(&in_boxes)) { continue; }
                let bj = in_boxes[j];
                let minx2 = bj.x0y0 & 0xFFFFu;
                let miny2 = bj.x0y0 >> 16u;
                let maxx2 = bj.x1y1 & 0xFFFFu;
                let maxy2 = bj.x1y1 >> 16u;
                if (!color_close(b.color565, bj.color565, params.color_tol)) { continue; }
                let overlap_x = overlap_1d(minx, maxx, minx2, maxx2);
                let overlap_y = overlap_1d(miny, maxy, miny2, maxy2);
                let gap_x = gap_1d(minx, maxx, minx2, maxx2);
                let gap_y = gap_1d(miny, maxy, miny2, maxy2);
                let merge_ok = (gap_x <= params.gap_x && overlap_y >= params.overlap_min) ||
                               (gap_y <= params.gap_y && overlap_x >= params.overlap_min);
                if (merge_ok) {
                    let li = atomicLoad(&labels[idx]);
                    let lj = atomicLoad(&labels[j]);
                    let mn = min(li, lj);
                    atomicMin(&labels[idx], mn);
                    atomicMin(&labels[j], mn);
                }
            }
        }
    }
}
