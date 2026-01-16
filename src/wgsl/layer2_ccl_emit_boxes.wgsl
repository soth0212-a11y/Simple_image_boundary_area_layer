struct Params {
    total_segments: u32,
    max_out: u32,
    min_w: u32,
    min_h: u32,
}

struct Segment {
    x0: u32,
    x1: u32,
    y_color: u32,
    pad: u32,
}

struct OutBox {
    x0y0: u32,
    x1y1: u32,
    color565: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read> segments: array<Segment>;
@group(0) @binding(1) var<storage, read> parent: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> bbox_minx: array<u32>;
@group(0) @binding(3) var<storage, read> bbox_miny: array<u32>;
@group(0) @binding(4) var<storage, read> bbox_maxx: array<u32>;
@group(0) @binding(5) var<storage, read> bbox_maxy: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> out_boxes: array<OutBox>;
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_segments) { return; }
    if (idx >= arrayLength(&parent)) { return; }
    if (atomicLoad(&parent[idx]) != idx) { return; }
    if (idx >= arrayLength(&bbox_minx)) { return; }
    let minx = bbox_minx[idx];
    let miny = bbox_miny[idx];
    let maxx = bbox_maxx[idx];
    let maxy = bbox_maxy[idx];
    if (minx == 0xFFFFFFFFu) { return; }
    if (maxx <= minx || maxy <= miny) { return; }
    let w = maxx - minx;
    let h = maxy - miny;
    if (w < params.min_w || h < params.min_h) { return; }
    if (idx >= arrayLength(&segments)) { return; }
    let color565 = segments[idx].y_color & 0xFFFFu;
    if (arrayLength(&out_count) == 0u) { return; }
    let out_idx = atomicAdd(&out_count[0], 1u);
    if (out_idx >= params.max_out || out_idx >= arrayLength(&out_boxes)) { return; }
    let x0y0 = (miny << 16u) | (minx & 0xFFFFu);
    let x1y1 = (maxy << 16u) | (maxx & 0xFFFFu);
    out_boxes[out_idx] = OutBox(x0y0, x1y1, color565, 0u);
}
