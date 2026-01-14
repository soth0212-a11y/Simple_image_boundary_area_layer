@group(0) @binding(0) var<storage, read> flat_boxes: array<u32>;
@group(0) @binding(1) var<storage, read_write> flat_count: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> label: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> out_boxes: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> out_valid: array<atomic<u32>>;

fn find_root(idx: u32) -> u32 {
    var r: u32 = idx;
    for (var step: u32 = 0u; step < 64u; step = step + 1u) {
        if (r >= arrayLength(&label)) { break; }
        let p: u32 = atomicLoad(&label[r]);
        if (p == r) { break; }
        r = p;
    }
    return r;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    let count: u32 = atomicLoad(&flat_count);
    if (idx >= count) { return; }
    if (idx * 2u + 1u >= arrayLength(&flat_boxes)) { return; }

    let b0: u32 = flat_boxes[idx * 2u];
    let b1: u32 = flat_boxes[idx * 2u + 1u];
    let x0: u32 = b0 & 0xFFFFu;
    let y0: u32 = b0 >> 16u;
    let x1: u32 = b1 & 0xFFFFu;
    let y1: u32 = b1 >> 16u;
    if (x1 <= x0 || y1 <= y0) { return; }

    let root: u32 = find_root(idx);
    let out_idx: u32 = root * 2u;
    if (out_idx + 1u >= arrayLength(&out_boxes) || root >= arrayLength(&out_valid)) { return; }

    var old_min: u32 = atomicLoad(&out_boxes[out_idx]);
    loop {
        let old_x: u32 = old_min & 0xFFFFu;
        let old_y: u32 = old_min >> 16u;
        let new_x: u32 = min(old_x, x0);
        let new_y: u32 = min(old_y, y0);
        let packed: u32 = (new_x & 0xFFFFu) | ((new_y & 0xFFFFu) << 16u);
        if (packed == old_min) { break; }
        let res = atomicCompareExchangeWeak(&out_boxes[out_idx], old_min, packed);
        if (res.exchanged) { break; }
        old_min = res.old_value;
    }

    var old_max: u32 = atomicLoad(&out_boxes[out_idx + 1u]);
    loop {
        let old_x: u32 = old_max & 0xFFFFu;
        let old_y: u32 = old_max >> 16u;
        let new_x: u32 = max(old_x, x1);
        let new_y: u32 = max(old_y, y1);
        let packed: u32 = (new_x & 0xFFFFu) | ((new_y & 0xFFFFu) << 16u);
        if (packed == old_max) { break; }
        let res = atomicCompareExchangeWeak(&out_boxes[out_idx + 1u], old_max, packed);
        if (res.exchanged) { break; }
        old_max = res.old_value;
    }
    atomicStore(&out_valid[root], 1u);
}
