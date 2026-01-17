struct Params {
    w: u32,
    h: u32,
    color_tol: u32,
    _pad1: u32,
}

struct Segment {
    tl: u32,
    br: u32,
    color565: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> s_active: array<u32>;
@group(0) @binding(1) var<storage, read> color565: array<u32>;
@group(0) @binding(2) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> segments: array<Segment>;
@group(0) @binding(4) var<uniform> params: Params;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

fn load_active(x: u32, y: u32, w: u32) -> u32 {
    let idx = idx2(x, y, w);
    return select(0u, s_active[idx] & 1u, idx < arrayLength(&s_active));
}

fn load_color565(x: u32, y: u32, w: u32) -> u32 {
    let idx = idx2(x, y, w);
    return select(0u, color565[idx] & 0xFFFFu, idx < arrayLength(&color565));
}

fn absdiff(a: u32, b: u32) -> u32 {
    return select(a - b, b - a, a >= b);
}

fn color_close_565(a: u32, b: u32, tol: u32) -> bool {
    let ar = (a >> 11u) & 31u;
    let ag = (a >> 5u) & 63u;
    let ab = a & 31u;
    let br = (b >> 11u) & 31u;
    let bg = (b >> 5u) & 63u;
    let bb = b & 31u;
    return absdiff(ar, br) <= tol && absdiff(ag, bg) <= (tol * 2u) && absdiff(ab, bb) <= tol;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    let tol = params.color_tol;
    let y = gid.x;
    if (w == 0u || h == 0u) { return; }
    if (y >= h) { return; }
    if (y + 1u >= arrayLength(&row_offsets)) { return; }
    if ((y & 1u) != 0u) { return; }
    var write = row_offsets[y];
    let row_end = row_offsets[y + 1u];
    for (var x: u32 = 0u; x < w; x = x + 2u) {
        let a = load_active(x, y, w);
        if (a == 0u) { continue; }
        let c = load_color565(x, y, w);

        let has_w = select(false, color_close_565(c, load_color565(x - 1u, y, w), tol), x > 0u);
        let has_e = select(false, color_close_565(c, load_color565(x + 1u, y, w), tol), x + 1u < w);
        let has_n = select(false, color_close_565(c, load_color565(x, y - 1u, w), tol), y > 0u);
        let has_s = select(false, color_close_565(c, load_color565(x, y + 1u, w), tol), y + 1u < h);

        let x0 = select(x, x - 1u, has_w && x > 0u);
        let x1 = select(x + 1u, min(x + 2u, w), has_e && (x + 1u < w));
        let y0 = select(y, y - 1u, has_n && y > 0u);
        let y1 = select(y + 1u, min(y + 2u, h), has_s && (y + 1u < h));
        let tl = (y0 << 16u) | (x0 & 0xFFFFu);
        let br = (y1 << 16u) | (x1 & 0xFFFFu);
        if (write < row_end && write < arrayLength(&segments)) {
            segments[write] = Segment(tl, br, (c & 0xFFFFu), 0u);
            write = write + 1u;
        }
    }
}
