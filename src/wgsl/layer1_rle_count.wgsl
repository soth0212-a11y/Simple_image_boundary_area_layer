struct Params {
    w: u32,
    h: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> s_active: array<u32>;
@group(0) @binding(1) var<storage, read> color565: array<u32>;
@group(0) @binding(2) var<storage, read_write> row_counts: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    let y = gid.x;
    if (w == 0u || h == 0u) { return; }
    if (y >= h) { return; }
    var count: u32 = 0u;
    var prev_a: u32 = 0u;
    var prev_c: u32 = 0u;
    for (var x: u32 = 0u; x < w; x = x + 1u) {
        let idx = idx2(x, y, w);
        let a = select(0u, s_active[idx] & 1u, idx < arrayLength(&s_active));
        let c = select(0u, color565[idx] & 0xFFFFu, idx < arrayLength(&color565));
        let seg_start = (a != 0u) && (x == 0u || prev_a == 0u || prev_c != c);
        if (seg_start) {
            count = count + 1u;
        }
        prev_a = a;
        prev_c = c;
    }
    if (y < arrayLength(&row_counts)) {
        row_counts[y] = count;
    }
}
