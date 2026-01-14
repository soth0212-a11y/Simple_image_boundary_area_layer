// L2 pooling stage (kernel=4, stride=2) for 1:1 L1 mask input.
// Outputs only bit0(INACTIVE) or bit1(ACTIVE) per pooled cell.

const KERNEL: u32 = 4u;
const STRIDE: u32 = 2u;
const TH_ACTIVE: u32 = 5u;
const KERNEL_AREA: u32 = KERNEL * KERNEL;


@group(0) @binding(0) var<storage, read> input_img_info: array<u32>; // [height, width]
@group(0) @binding(1) var<storage, read> layer1_out_mask: array<u32>; // width*height, bit0/bit1
@group(0) @binding(2) var<storage, read_write> pooled_mask: array<u32>; // out_w*out_h

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&input_img_info) < 2u) { return; }
    let height: u32 = input_img_info[0];
    let width: u32 = input_img_info[1];
    if (width == 0u || height == 0u) { return; }

    var out_w: u32 = 1u;
    var out_h: u32 = 1u;
    if (width < KERNEL || height < KERNEL) {
        out_w = 1u;
        out_h = 1u;
    } else {
        out_w = ((width + STRIDE - 1u - KERNEL) / STRIDE) + 1u;
        out_h = ((height + STRIDE - 1u - KERNEL) / STRIDE) + 1u;
    }
    if (out_w == 0u || out_h == 0u) { return; }

    let px: u32 = gid.x;
    let py: u32 = gid.y;
    if (px >= out_w || py >= out_h) { return; }

    let out_idx: u32 = py * out_w + px;
    if (out_idx >= arrayLength(&pooled_mask)) { return; }

    var active_cnt: u32 = 0u;
    var inactive_cnt: u32 = 0u;
    let sx: u32 = px * STRIDE;
    let sy: u32 = py * STRIDE;
    for (var dy: u32 = 0u; dy < KERNEL; dy = dy + 1u) {
        let iy: u32 = sy + dy;
        if (iy >= height) { continue; }
        for (var dx: u32 = 0u; dx < KERNEL; dx = dx + 1u) {
            let ix: u32 = sx + dx;
            if (ix >= width) { continue; }
            let idx: u32 = iy * width + ix;
            if (idx >= arrayLength(&layer1_out_mask)) { continue; }
            let m: u32 = layer1_out_mask[idx];
            active_cnt = active_cnt + ((m >> 1u) & 1u);
            inactive_cnt = inactive_cnt + (m & 1u);
        }
    }

    // Majority pooling: active wins ties.
    let pooled_active: bool = active_cnt >= inactive_cnt;
    pooled_mask[out_idx] = select(1u << 0u, 1u << 1u, pooled_active);
}
