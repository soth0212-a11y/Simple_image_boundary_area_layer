struct OffsetParams {
    offset_x: i32,
    offset_y: i32,
}

@group(0) @binding(0) var<storage, read> input_img_info: array<u32>;
@group(0) @binding(1) var<storage, read> flat_boxes: array<u32>;
@group(0) @binding(2) var<storage, read_write> flat_count: atomic<u32>;
@group(0) @binding(3) var<storage, read> params: OffsetParams;
@group(0) @binding(4) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> cell_items: array<u32>;
@group(0) @binding(6) var<storage, read_write> label: array<atomic<u32>>;

const STRIDE: u32 = 41u;
const STRIDE_I: i32 = 41;
const KMAX: u32 = 16u;
const R: u32 = 16u;
const R2: u32 = R * R;

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

fn abs_i32(v: i32) -> u32 {
    return u32(select(v, -v, v < 0));
}

fn center_for(idx: u32) -> vec2<u32> {
    let bbox_idx: u32 = idx * 2u;
    let b0: u32 = flat_boxes[bbox_idx];
    let b1: u32 = flat_boxes[bbox_idx + 1u];
    let x0: u32 = b0 & 0xFFFFu;
    let y0: u32 = b0 >> 16u;
    let x1: u32 = b1 & 0xFFFFu;
    let y1: u32 = b1 >> 16u;
    return vec2<u32>((x0 + x1) / 2u, (y0 + y1) / 2u);
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

    let center_i: vec2<u32> = center_for(idx);
    let sx: i32 = (i32(center_i.x) - params.offset_x) / STRIDE_I;
    let sy: i32 = (i32(center_i.y) - params.offset_y) / STRIDE_I;
    let cell_x: i32 = clamp(sx, 0, i32(grid.x) - 1);
    let cell_y: i32 = clamp(sy, 0, i32(grid.y) - 1);

    for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
        let ny: i32 = cell_y + oy;
        if (ny < 0 || ny >= i32(grid.y)) { continue; }
        for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
            let nx: i32 = cell_x + ox;
            if (nx < 0 || nx >= i32(grid.x)) { continue; }
            let neighbor_id: u32 = u32(ny) * grid.x + u32(nx);
            if (neighbor_id >= arrayLength(&cell_counts)) { continue; }
            let n_count: u32 = atomicLoad(&cell_counts[neighbor_id]);
            let limit: u32 = min(n_count, KMAX);
            for (var t: u32 = 0u; t < limit; t = t + 1u) {
                let item_idx: u32 = neighbor_id * KMAX + t;
                if (item_idx >= arrayLength(&cell_items)) { break; }
                let j: u32 = cell_items[item_idx];
                if (j == idx || j >= count) { continue; }
                let center_j: vec2<u32> = center_for(j);
                let dx: i32 = i32(center_j.x) - i32(center_i.x);
                let dy: i32 = i32(center_j.y) - i32(center_i.y);
                let dx_u: u32 = abs_i32(dx);
                let dy_u: u32 = abs_i32(dy);
                if (dx_u > R || dy_u > R) { continue; }
                let dist2: u32 = dx_u * dx_u + dy_u * dy_u;
                if (dist2 <= R2) {
                    let li: u32 = atomicLoad(&label[idx]);
                    let lj: u32 = atomicLoad(&label[j]);
                    let new_label: u32 = min(li, lj);
                    atomicMin(&label[idx], new_label);
                    atomicMin(&label[j], new_label);
                }
            }
        }
    }

    for (var step: u32 = 0u; step < 2u; step = step + 1u) {
        let li: u32 = atomicLoad(&label[idx]);
        let lli: u32 = atomicLoad(&label[li]);
        atomicStore(&label[idx], lli);
    }
}
