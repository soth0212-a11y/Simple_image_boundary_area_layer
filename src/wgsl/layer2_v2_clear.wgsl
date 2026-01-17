struct Params {
    bins_count: u32,
    total_boxes: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> bin_count: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> bin_cursor: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> comp_minx: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> comp_miny: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> comp_maxx: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> comp_maxy: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> out_count: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.bins_count) {
        if (idx < arrayLength(&bin_count)) { atomicStore(&bin_count[idx], 0u); }
        if (idx < arrayLength(&bin_cursor)) { atomicStore(&bin_cursor[idx], 0u); }
    }
    if (idx < params.total_boxes) {
        if (idx < arrayLength(&comp_minx)) { atomicStore(&comp_minx[idx], UINT_MAX); }
        if (idx < arrayLength(&comp_miny)) { atomicStore(&comp_miny[idx], UINT_MAX); }
        if (idx < arrayLength(&comp_maxx)) { atomicStore(&comp_maxx[idx], 0u); }
        if (idx < arrayLength(&comp_maxy)) { atomicStore(&comp_maxy[idx], 0u); }
    }
    if (idx == 0u && arrayLength(&out_count) > 0u) {
        atomicStore(&out_count[0], 0u);
    }
}
