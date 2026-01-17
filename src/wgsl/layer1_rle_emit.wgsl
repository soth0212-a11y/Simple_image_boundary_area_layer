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

fn emit_bbox(
    write: ptr<function, u32>,
    row_end: u32,
    w: u32,
    h: u32,
    x: u32,
    y: u32,
    out_c: u32,
    has_w: bool,
    has_e: bool,
    has_n: bool,
    has_s: bool,
) {
    // Base: center cell.
    var x0 = x;
    var x1 = min(x + 1u, w);
    var y0 = y;
    var y1 = min(y + 1u, h);

    // Color-based expansion attempt (+1) per direction.
    if (has_w && x0 > 0u) { x0 = x0 - 1u; }
    if (has_e && x1 < w) { x1 = min(x1 + 1u, w); }
    if (has_n && y0 > 0u) { y0 = y0 - 1u; }
    if (has_s && y1 < h) { y1 = min(y1 + 1u, h); }

    if (*write < row_end && *write < arrayLength(&segments)) {
        let tl = (y0 << 16u) | (x0 & 0xFFFFu);
        let br = (y1 << 16u) | (x1 & 0xFFFFu);
        segments[*write] = Segment(tl, br, (out_c & 0xFFFFu), 0u);
        *write = *write + 1u;
    }
}

fn band_key_565(c: u32) -> u32 {
    // Coarse banding (deterministic): r5>>2, g6>>3, b5>>2 => 3 bits each (0..7).
    let r = (c >> 11u) & 31u;
    let g = (c >> 5u) & 63u;
    let b = c & 31u;
    let r3 = r >> 2u;
    let g3 = g >> 3u;
    let b3 = b >> 2u;
    return (r3 << 6u) | (g3 << 3u) | b3;
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
    var write = row_offsets[y];
    let row_end = row_offsets[y + 1u];
    for (var x: u32 = 0u; x < w; x = x + 1u) {
        let a = load_active(x, y, w);
        if (a == 0u) { continue; }
        let center_c = load_color565(x, y, w);

        // Color-based expansion attempt.
        let has_w = select(false, color_close_565(center_c, load_color565(x - 1u, y, w), tol), x > 0u);
        let has_e = select(false, color_close_565(center_c, load_color565(x + 1u, y, w), tol), x + 1u < w);
        let has_n = select(false, color_close_565(center_c, load_color565(x, y - 1u, w), tol), y > 0u);
        let has_s = select(false, color_close_565(center_c, load_color565(x, y + 1u, w), tol), y + 1u < h);

        // Representative color: among inactive N/E/S/W, pick most frequent band; ties -> smaller bandKey.
        // Then pick the smallest q565 within that band (deterministic). If no inactive neighbors, use center color.
        var n_inactive: bool = false;
        var e_inactive: bool = false;
        var s_inactive: bool = false;
        var w_inactive: bool = false;
        var cn: u32 = 0u;
        var ce: u32 = 0u;
        var cs: u32 = 0u;
        var cw: u32 = 0u;

        if (y > 0u) {
            n_inactive = load_active(x, y - 1u, w) == 0u;
            if (n_inactive) { cn = load_color565(x, y - 1u, w); }
        }
        if (x + 1u < w) {
            e_inactive = load_active(x + 1u, y, w) == 0u;
            if (e_inactive) { ce = load_color565(x + 1u, y, w); }
        }
        if (y + 1u < h) {
            s_inactive = load_active(x, y + 1u, w) == 0u;
            if (s_inactive) { cs = load_color565(x, y + 1u, w); }
        }
        if (x > 0u) {
            w_inactive = load_active(x - 1u, y, w) == 0u;
            if (w_inactive) { cw = load_color565(x - 1u, y, w); }
        }

        var out_c: u32 = center_c;
        var best_key: u32 = 0u;
        var best_count: u32 = 0u;
        var best_color: u32 = 0u;
        var has_any: bool = false;

        // Small fixed set (<=4), brute-force counts is fine.
        if (n_inactive) {
            let kn = band_key_565(cn);
            var cnt: u32 = 1u;
            if (e_inactive && band_key_565(ce) == kn) { cnt = cnt + 1u; }
            if (s_inactive && band_key_565(cs) == kn) { cnt = cnt + 1u; }
            if (w_inactive && band_key_565(cw) == kn) { cnt = cnt + 1u; }
            var minc: u32 = cn;
            if (e_inactive && band_key_565(ce) == kn) { minc = min(minc, ce); }
            if (s_inactive && band_key_565(cs) == kn) { minc = min(minc, cs); }
            if (w_inactive && band_key_565(cw) == kn) { minc = min(minc, cw); }
            if (!has_any || cnt > best_count || (cnt == best_count && kn < best_key)) {
                has_any = true;
                best_key = kn;
                best_count = cnt;
                best_color = minc;
            }
        }
        if (e_inactive) {
            let ke = band_key_565(ce);
            var cnt: u32 = 1u;
            if (n_inactive && band_key_565(cn) == ke) { cnt = cnt + 1u; }
            if (s_inactive && band_key_565(cs) == ke) { cnt = cnt + 1u; }
            if (w_inactive && band_key_565(cw) == ke) { cnt = cnt + 1u; }
            var minc: u32 = ce;
            if (n_inactive && band_key_565(cn) == ke) { minc = min(minc, cn); }
            if (s_inactive && band_key_565(cs) == ke) { minc = min(minc, cs); }
            if (w_inactive && band_key_565(cw) == ke) { minc = min(minc, cw); }
            if (!has_any || cnt > best_count || (cnt == best_count && ke < best_key)) {
                has_any = true;
                best_key = ke;
                best_count = cnt;
                best_color = minc;
            }
        }
        if (s_inactive) {
            let ks = band_key_565(cs);
            var cnt: u32 = 1u;
            if (n_inactive && band_key_565(cn) == ks) { cnt = cnt + 1u; }
            if (e_inactive && band_key_565(ce) == ks) { cnt = cnt + 1u; }
            if (w_inactive && band_key_565(cw) == ks) { cnt = cnt + 1u; }
            var minc: u32 = cs;
            if (n_inactive && band_key_565(cn) == ks) { minc = min(minc, cn); }
            if (e_inactive && band_key_565(ce) == ks) { minc = min(minc, ce); }
            if (w_inactive && band_key_565(cw) == ks) { minc = min(minc, cw); }
            if (!has_any || cnt > best_count || (cnt == best_count && ks < best_key)) {
                has_any = true;
                best_key = ks;
                best_count = cnt;
                best_color = minc;
            }
        }
        if (w_inactive) {
            let kw = band_key_565(cw);
            var cnt: u32 = 1u;
            if (n_inactive && band_key_565(cn) == kw) { cnt = cnt + 1u; }
            if (e_inactive && band_key_565(ce) == kw) { cnt = cnt + 1u; }
            if (s_inactive && band_key_565(cs) == kw) { cnt = cnt + 1u; }
            var minc: u32 = cw;
            if (n_inactive && band_key_565(cn) == kw) { minc = min(minc, cn); }
            if (e_inactive && band_key_565(ce) == kw) { minc = min(minc, ce); }
            if (s_inactive && band_key_565(cs) == kw) { minc = min(minc, cs); }
            if (!has_any || cnt > best_count || (cnt == best_count && kw < best_key)) {
                has_any = true;
                best_key = kw;
                best_count = cnt;
                best_color = minc;
            }
        }
        if (has_any) {
            out_c = best_color;
        }

        emit_bbox(&write, row_end, w, h, x, y, out_c, has_w, has_e, has_n, has_s);
    }
}
