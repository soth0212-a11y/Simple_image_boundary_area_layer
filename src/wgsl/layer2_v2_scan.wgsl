struct Params {
    bins_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> bin_count: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> bin_offset: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var sum: u32 = 0u;
    let n = min(params.bins_count, arrayLength(&bin_count));
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        if (i < arrayLength(&bin_offset)) {
            bin_offset[i] = sum;
        }
        sum = sum + atomicLoad(&bin_count[i]);
    }
    if (n < arrayLength(&bin_offset)) {
        bin_offset[n] = sum;
    }
}
