struct Params {
    groups_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> group_count: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> group_minx: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> group_miny: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> group_maxx: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> group_maxy: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> group_color: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> out_count: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.groups_count) {
        if (idx < arrayLength(&group_count)) { atomicStore(&group_count[idx], 0u); }
        if (idx < arrayLength(&group_minx)) { atomicStore(&group_minx[idx], UINT_MAX); }
        if (idx < arrayLength(&group_miny)) { atomicStore(&group_miny[idx], UINT_MAX); }
        if (idx < arrayLength(&group_maxx)) { atomicStore(&group_maxx[idx], 0u); }
        if (idx < arrayLength(&group_maxy)) { atomicStore(&group_maxy[idx], 0u); }
        if (idx < arrayLength(&group_color)) { atomicStore(&group_color[idx], UINT_MAX); }
    }
    if (idx == 0u && arrayLength(&out_count) > 0u) {
        atomicStore(&out_count[0], 0u);
    }
}

