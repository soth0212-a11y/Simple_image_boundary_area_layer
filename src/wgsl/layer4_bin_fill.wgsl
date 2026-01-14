struct OffsetParams {
    offset_x: i32,
    offset_y: i32,
}

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>;
@group(0) @binding(1) var<storage, read> flat_boxes: array<u32>;
@group(0) @binding(2) var<storage, read_write> flat_count: atomic<u32>;
@group(0) @binding(3) var<storage, read> params: OffsetParams;
@group(0) @binding(4) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cell_items: array<u32>;
@group(0) @binding(6) var<storage, read_write> overflow_flags: array<atomic<u32>>;

const STRIDE: u32 = 41u;
const STRIDE_I: i32 = 41;
const KMAX: u32 = 16u;

fn compute_out_dims(height: u32, width: u32) -> vec2<u32> {
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

fn compute_grid_dims(out_w: u32, out_h: u32) -> vec2<u32> {
    let grid_w: u32 = (out_w + STRIDE - 1u) / STRIDE;
    let grid_h: u32 = (out_h + STRIDE - 1u) / STRIDE;
    return vec2<u32>(grid_w, grid_h);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    let count: u32 = atomicLoad(&flat_count);
    if (idx >= count) { return; }
    if (arrayLength(&input_img_info) < 2u) { return; }

    let height: u32 = input_img_info[0];
    let width: u32 = input_img_info[1];
    if (width == 0u || height == 0u) { return; }

    let dims: vec2<u32> = compute_out_dims(height, width);
    let grid: vec2<u32> = compute_grid_dims(dims.x, dims.y);
    if (grid.x == 0u || grid.y == 0u) { return; }

    let bbox_idx: u32 = idx * 2u;
    if (bbox_idx + 1u >= arrayLength(&flat_boxes)) { return; }
    let b0: u32 = flat_boxes[bbox_idx];
    let b1: u32 = flat_boxes[bbox_idx + 1u];
    let x0: u32 = b0 & 0xFFFFu;
    let y0: u32 = b0 >> 16u;
    let x1: u32 = b1 & 0xFFFFu;
    let y1: u32 = b1 >> 16u;
    let cx: u32 = (x0 + x1) / 2u;
    let cy: u32 = (y0 + y1) / 2u;

    let sx: i32 = (i32(cx) - params.offset_x) / STRIDE_I;
    let sy: i32 = (i32(cy) - params.offset_y) / STRIDE_I;
    let gx: i32 = clamp(sx, 0, i32(grid.x) - 1);
    let gy: i32 = clamp(sy, 0, i32(grid.y) - 1);
    let cell_id: u32 = u32(gy) * grid.x + u32(gx);
    if (cell_id >= arrayLength(&cell_counts)) { return; }

    let slot: u32 = atomicAdd(&cell_counts[cell_id], 1u);
    if (slot < KMAX) {
        let item_idx: u32 = cell_id * KMAX + slot;
        if (item_idx < arrayLength(&cell_items)) {
            cell_items[item_idx] = idx;
        }
    } else {
        atomicStore(&overflow_flags[cell_id], 1u);
    }
}
