struct Params {
    groups_count: u32,
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

@group(0) @binding(0) var<storage, read> group_count: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> group_minx: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> group_miny: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> group_maxx: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> group_maxy: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> group_color: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> out_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> out_boxes: array<OutBox>;
@group(0) @binding(8) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gid0 = gid.x;
    if (gid0 >= params.groups_count) { return; }
    if (gid0 >= arrayLength(&group_count)) { return; }
    let cnt = atomicLoad(&group_count[gid0]);
    if (cnt == 0u) { return; }
    let minx = atomicLoad(&group_minx[gid0]);
    if (minx == UINT_MAX) { return; }
    let miny = atomicLoad(&group_miny[gid0]);
    let maxx = atomicLoad(&group_maxx[gid0]);
    let maxy = atomicLoad(&group_maxy[gid0]);
    if (maxx <= minx || maxy <= miny) { return; }
    if (arrayLength(&out_count) == 0u) { return; }
    let out_idx = atomicAdd(&out_count[0], 1u);
    if (out_idx >= arrayLength(&out_boxes)) { return; }
    let x0y0 = (miny << 16u) | (minx & 0xFFFFu);
    let x1y1 = (maxy << 16u) | (maxx & 0xFFFFu);
    let c = atomicLoad(&group_color[gid0]);
    // Output contract (L2 v3):
    // - color565: representative q565 (16-bit)
    // - flags: group_id (= bandKey*bins_count + bin_id)
    out_boxes[out_idx] = OutBox(x0y0, x1y1, c & 0xFFFFu, gid0);
}
