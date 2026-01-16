struct Params {
    w: u32,
    h: u32,
    stride: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> s_active: array<u32>;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> bbox0: array<u32>;
@group(0) @binding(3) var<storage, read_write> bbox1: array<u32>;
@group(0) @binding(4) var<storage, read_write> bbox_color: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

fn active_at(x: i32, y: i32, w: u32, h: u32) -> u32 {
    if (x < 0 || y < 0) { return 0u; }
    let ux: u32 = u32(x);
    let uy: u32 = u32(y);
    if (ux >= w || uy >= h) { return 0u; }
    let idx = idx2(ux, uy, w);
    if (idx >= arrayLength(&s_active)) { return 0u; }
    return s_active[idx] & 1u;
}

fn any_neighbor_active(cx: u32, cy: u32, w: u32, h: u32) -> bool {
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            if (active_at(i32(cx) + dx, i32(cy) + dy, w, h) != 0u) {
                return true;
            }
        }
    }
    return false;
}

fn write_empty(out_idx: u32) {
    if (out_idx < arrayLength(&bbox0)) { bbox0[out_idx] = UINT_MAX; }
    if (out_idx < arrayLength(&bbox1)) { bbox1[out_idx] = UINT_MAX; }
    if (out_idx < arrayLength(&bbox_color)) { bbox_color[out_idx] = 0u; }
}

fn write_bbox(out_idx: u32, x0: u32, y0: u32, x1: u32, y1: u32, color: u32) {
    if (out_idx < arrayLength(&bbox0)) { bbox0[out_idx] = (x0 & 0xFFFFu) | ((y0 & 0xFFFFu) << 16u); }
    if (out_idx < arrayLength(&bbox1)) { bbox1[out_idx] = (x1 & 0xFFFFu) | ((y1 & 0xFFFFu) << 16u); }
    if (out_idx < arrayLength(&bbox_color)) { bbox_color[out_idx] = color; }
}

fn packed_rgb565_at(idx: u32) -> u32 {
    let base = idx * 2u;
    if (base >= arrayLength(&packed)) { return 0u; }
    let flags = packed[base];
    let q565 = (flags >> 9u) & 0xFFFFu;
    return q565;
}

@compute @workgroup_size(16, 16, 1)
fn l1_bbox_stride1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&s_active)) { return; }
    if ((s_active[idx] & 1u) == 0u) {
        write_empty(idx);
        return;
    }
    let color = packed_rgb565_at(idx);
    var x0: u32 = x;
    var y0: u32 = y;
    var x1: u32 = x + 1u;
    var y1: u32 = y + 1u;
    if (any_neighbor_active(x, y, w, h)) {
        x0 = select(0u, x - 1u, x > 0u);
        y0 = select(0u, y - 1u, y > 0u);
        x1 = min(w, x + 2u);
        y1 = min(h, y + 2u);
    }
    write_bbox(idx, x0, y0, x1, y1, color);
}

@compute @workgroup_size(16, 16, 1)
fn l1_bbox_stride2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    let stride = params.stride;
    if (w == 0u || h == 0u || stride == 0u) { return; }
    let out_w = (w + stride - 1u) / stride;
    let out_h = (h + stride - 1u) / stride;
    let ox = gid.x;
    let oy = gid.y;
    if (ox >= out_w || oy >= out_h) { return; }
    let base_x = ox * stride;
    let base_y = oy * stride;
    if (base_x >= w || base_y >= h) {
        write_empty(oy * out_w + ox);
        return;
    }
    let idx = idx2(base_x, base_y, w);
    let out_idx = oy * out_w + ox;
    if (idx >= arrayLength(&s_active)) { return; }
    if ((s_active[idx] & 1u) == 0u) {
        write_empty(out_idx);
        return;
    }
    let color = packed_rgb565_at(idx);
    var x0: u32 = base_x;
    var y0: u32 = base_y;
    var x1: u32 = base_x + 1u;
    var y1: u32 = base_y + 1u;
    if (any_neighbor_active(base_x, base_y, w, h)) {
        x0 = select(0u, base_x - 1u, base_x > 0u);
        y0 = select(0u, base_y - 1u, base_y > 0u);
        x1 = min(w, base_x + 2u);
        y1 = min(h, base_y + 2u);
    }
    write_bbox(out_idx, x0, y0, x1, y1, color);
}
