struct Params {
    total_boxes: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> labels: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_boxes) { return; }
    if (idx >= arrayLength(&labels)) { return; }
    let l = atomicLoad(&labels[idx]);
    if (l < arrayLength(&labels)) {
        let l2 = atomicLoad(&labels[l]);
        atomicStore(&labels[idx], l2);
    }
}
