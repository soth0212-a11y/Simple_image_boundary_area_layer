struct Params {
    total_boxes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct OutBox {
    x0y0: u32,
    x1y1: u32,
    color565: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> in_boxes: array<OutBox>;
@group(0) @binding(1) var<storage, read> labels: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> comp_minx: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> comp_miny: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> comp_maxx: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> comp_maxy: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> out_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> out_boxes: array<OutBox>;
@group(0) @binding(8) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_boxes) { return; }
    if (idx >= arrayLength(&labels) || idx >= arrayLength(&in_boxes)) { return; }
    if (atomicLoad(&labels[idx]) != idx) { return; }
    if (idx >= arrayLength(&comp_minx)) { return; }
    let minx = atomicLoad(&comp_minx[idx]);
    if (minx == UINT_MAX) { return; }
    let miny = atomicLoad(&comp_miny[idx]);
    let maxx = atomicLoad(&comp_maxx[idx]);
    let maxy = atomicLoad(&comp_maxy[idx]);
    if (maxx <= minx || maxy <= miny) { return; }
    if (arrayLength(&out_count) == 0u) { return; }
    let out_idx = atomicAdd(&out_count[0], 1u);
    if (out_idx >= arrayLength(&out_boxes)) { return; }
    let x0y0 = (miny << 16u) | (minx & 0xFFFFu);
    let x1y1 = (maxy << 16u) | (maxx & 0xFFFFu);
    let color565 = in_boxes[idx].color565 & 0xFFFFu;
    out_boxes[out_idx] = OutBox(x0y0, x1y1, color565, 0u);
}
