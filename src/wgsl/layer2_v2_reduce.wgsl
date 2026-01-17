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
@group(0) @binding(2) var<storage, read_write> comp_minx: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> comp_miny: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> comp_maxx: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> comp_maxy: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_boxes) { return; }
    if (idx >= arrayLength(&in_boxes) || idx >= arrayLength(&labels)) { return; }
    let label = atomicLoad(&labels[idx]);
    if (label >= arrayLength(&comp_minx)) { return; }
    let b = in_boxes[idx];
    let minx = b.x0y0 & 0xFFFFu;
    let miny = b.x0y0 >> 16u;
    let maxx = b.x1y1 & 0xFFFFu;
    let maxy = b.x1y1 >> 16u;
    atomicMin(&comp_minx[label], minx);
    atomicMin(&comp_miny[label], miny);
    atomicMax(&comp_maxx[label], maxx);
    atomicMax(&comp_maxy[label], maxy);
}
