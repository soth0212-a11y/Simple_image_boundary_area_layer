struct Params {
    total_segments: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Segment {
    x0: u32,
    x1: u32,
    y_color: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> segments: array<Segment>;
@group(0) @binding(1) var<storage, read_write> parent: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> bbox_minx: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bbox_miny: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> bbox_maxx: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> bbox_maxy: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;

fn load_parent(idx: u32) -> u32 {
    return atomicLoad(&parent[idx]);
}

fn store_parent(idx: u32, value: u32) {
    atomicStore(&parent[idx], value);
}

fn find(x_in: u32) -> u32 {
    var x = x_in;
    var p = load_parent(x);
    while (p != x) {
        let gp = load_parent(p);
        store_parent(x, gp);
        x = p;
        p = gp;
    }
    return x;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_segments) { return; }
    if (idx >= arrayLength(&segments) || idx >= arrayLength(&parent)) { return; }
    let r = find(idx);
    let seg = segments[idx];
    let x0 = seg.x0;
    let x1 = seg.x1;
    let y = seg.y_color >> 16u;
    if (r >= arrayLength(&bbox_minx)) { return; }
    atomicMin(&bbox_minx[r], x0);
    atomicMin(&bbox_miny[r], y);
    atomicMax(&bbox_maxx[r], x1);
    atomicMax(&bbox_maxy[r], y + 1u);
}
