// L2 pooling stage (kernel=2, stride=1) over 2x2-reduced L1 counts.
// Outputs only bit0(INACTIVE) or bit1(ACTIVE) per pooled cell.

const KERNEL: u32 = 2u;
const STRIDE: u32 = 1u;

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>; // [height, width]
@group(0) @binding(1) var<storage, read> layer0_out_packed: array<u32>; // vec2<u32> per cell (out0/out1)
@group(0) @binding(2) var<storage, read_write> pooled_mask: array<u32>; // out_w*out_h

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&input_img_info) < 2u) { return; }
    let height: u32 = input_img_info[0];
    let width: u32 = input_img_info[1];
    if (width == 0u || height == 0u) { return; }

    let l1_w: u32 = (width + 1u) / 2u;
    let l1_h: u32 = (height + 1u) / 2u;
    if (l1_w == 0u || l1_h == 0u) { return; }

    var out_w: u32 = 1u;
    var out_h: u32 = 1u;
    if (l1_w < KERNEL || l1_h < KERNEL) {
        out_w = 1u;
        out_h = 1u;
    } else {
        out_w = ((l1_w + STRIDE - 1u - KERNEL) / STRIDE) + 1u;
        out_h = ((l1_h + STRIDE - 1u - KERNEL) / STRIDE) + 1u;
    }
    if (out_w == 0u || out_h == 0u) { return; }

    let px: u32 = gid.x;
    let py: u32 = gid.y;
    if (px >= out_w || py >= out_h) { return; }

    let out_idx: u32 = py * out_w + px;
    if (out_idx >= arrayLength(&pooled_mask)) { return; }

    var active_cnt: u32 = 0u;
    var total_cnt: u32 = 0u;
    let sx: u32 = px * STRIDE;
    let sy: u32 = py * STRIDE;
    for (var dy: u32 = 0u; dy < KERNEL; dy = dy + 1u) {
        let iy: u32 = sy + dy;
        if (iy >= l1_h) { continue; }
        for (var dx: u32 = 0u; dx < KERNEL; dx = dx + 1u) {
            let ix: u32 = sx + dx;
            if (ix >= l1_w) { continue; }
            let idx: u32 = iy * l1_w + ix;
            let base: u32 = idx * 2u;
            if (base >= arrayLength(&layer0_out_packed)) { continue; }
            let packed0: u32 = layer0_out_packed[base];
            let cell_active: u32 = packed0 & 1u;
            active_cnt = active_cnt + cell_active;
            total_cnt = total_cnt + 1u;
        }
    }

    // Majority pooling: active wins ties.
    let inactive_total: u32 = total_cnt - active_cnt;
    let pooled_active: bool = active_cnt >= inactive_total;
    pooled_mask[out_idx] = select(1u << 0u, 1u << 1u, pooled_active);
}
