struct Params {
    w: u32,
    h: u32,
    color_tol: u32,
    gap_x: u32,
    gap_y: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Segment {
    x0: u32,
    x1: u32,
    y_color: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> segments: array<Segment>;
@group(0) @binding(2) var<storage, read_write> parent: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

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

fn union_sets(a: u32, b: u32) {
    var ra = find(a);
    var rb = find(b);
    loop {
        if (ra == rb) { return; }
        let hi = max(ra, rb);
        let lo = min(ra, rb);
        let old = atomicCompareExchangeWeak(&parent[hi], hi, lo);
        if (old.exchanged) { return; }
        ra = find(ra);
        rb = find(rb);
    }
}

fn absdiff(a: u32, b: u32) -> u32 {
    return select(a - b, b - a, a >= b);
}

fn color_close(ca: u32, cb: u32, tol: u32) -> bool {
    let ar = (ca >> 11u) & 31u;
    let ag = (ca >> 5u) & 63u;
    let ab = ca & 31u;
    let br = (cb >> 11u) & 31u;
    let bg = (cb >> 5u) & 63u;
    let bb = cb & 31u;
    return absdiff(ar, br) <= tol && absdiff(ag, bg) <= tol && absdiff(ab, bb) <= tol;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    let y = gid.x;
    if (w == 0u || h == 0u) { return; }
    let cur_start = row_offsets[y];
    let cur_end = row_offsets[y + 1u];
    if (cur_start >= cur_end) { return; }
    for (var dy: u32 = 1u; dy <= params.gap_y; dy = dy + 1u) {
        if (y < dy) { break; }
        let py = y - dy;
        if (py + 1u >= arrayLength(&row_offsets)) { continue; }
        let prev_start = row_offsets[py];
        let prev_end = row_offsets[py + 1u];
        var i = prev_start;
        var j = cur_start;
        while (i < prev_end && j < cur_end) {
            if (i >= arrayLength(&segments) || j >= arrayLength(&segments)) { break; }
            let sp = segments[i];
            let sc = segments[j];
            let x0p = sp.x0;
            let x1p = sp.x1;
            let x0c = sc.x0;
            let x1c = sc.x1;
            let cp = sp.y_color & 0xFFFFu;
            let cc = sc.y_color & 0xFFFFu;

            if (x1p + params.gap_x <= x0c) {
                i = i + 1u;
                continue;
            }
            if (x1c + params.gap_x <= x0p) {
                j = j + 1u;
                continue;
            }
            if (color_close(cp, cc, params.color_tol)) {
                let ox0 = max(x0p, x0c);
                let ox1 = min(x1p, x1c);
                let gap_ok = (x0c <= x1p + params.gap_x) && (x0p <= x1c + params.gap_x);
                if (ox0 < ox1 || gap_ok) {
                    union_sets(i, j);
                }
            }
            if (x1p < x1c) {
                i = i + 1u;
            } else {
                j = j + 1u;
            }
        }
    }
}
