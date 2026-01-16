struct Params {
    total_segments: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> parent: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> bbox_minx: array<u32>;
@group(0) @binding(2) var<storage, read_write> bbox_miny: array<u32>;
@group(0) @binding(3) var<storage, read_write> bbox_maxx: array<u32>;
@group(0) @binding(4) var<storage, read_write> bbox_maxy: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_count: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_segments) { return; }
    if (idx < arrayLength(&parent)) {
        atomicStore(&parent[idx], idx);
    }
    if (idx < arrayLength(&bbox_minx)) { bbox_minx[idx] = UINT_MAX; }
    if (idx < arrayLength(&bbox_miny)) { bbox_miny[idx] = UINT_MAX; }
    if (idx < arrayLength(&bbox_maxx)) { bbox_maxx[idx] = 0u; }
    if (idx < arrayLength(&bbox_maxy)) { bbox_maxy[idx] = 0u; }
    if (idx == 0u && arrayLength(&out_count) > 0u) {
        atomicStore(&out_count[0], 0u);
    }
}
