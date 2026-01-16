struct Params0 {
    w: u32,
    h: u32,
    th: u32,
    iter: u32,
}

@group(0) @binding(0) var<storage, read> cell_rgb: array<u32>;
@group(0) @binding(1) var<storage, read> edge4_in: array<u32>;
@group(0) @binding(2) var<storage, read> s_active: array<u32>;
@group(0) @binding(3) var<storage, read> plane_id_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> plane_id_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> edge4_out: array<u32>;
@group(0) @binding(6) var<uniform> params0: Params0;

const UINT_MAX: u32 = 0xFFFFFFFFu;

fn r(c: u32) -> u32 { return c & 255u; }
fn g(c: u32) -> u32 { return (c >> 8u) & 255u; }
fn b(c: u32) -> u32 { return (c >> 16u) & 255u; }

fn absdiff(a: u32, b: u32) -> u32 {
    return select(a - b, b - a, a >= b);
}

fn color_diff(ci: u32, cj: u32) -> u32 {
    return absdiff(r(ci), r(cj)) + absdiff(g(ci), g(cj)) + absdiff(b(ci), b(cj));
}

fn sameplane(idx: u32, nidx: u32) -> bool {
    if (idx >= arrayLength(&cell_rgb) || nidx >= arrayLength(&cell_rgb)) {
        return false;
    }
    let d = color_diff(cell_rgb[idx], cell_rgb[nidx]);
    return d <= params0.th;
}

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

@compute @workgroup_size(16, 16, 1)
fn l1_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params0.w;
    let h = params0.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&plane_id_out)) { return; }

    let act = select(0u, s_active[idx] & 1u, idx < arrayLength(&s_active));
    plane_id_out[idx] = select(idx, UINT_MAX, act != 0u);
    var edge: u32 = 0u;
    if (idx < arrayLength(&edge4_in)) {
        edge = edge4_in[idx] & 0xFu;
    }
    if (idx < arrayLength(&edge4_out)) { edge4_out[idx] = edge; }

}

@compute @workgroup_size(16, 16, 1)
fn l1_propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params0.w;
    let h = params0.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&plane_id_in) || idx >= arrayLength(&plane_id_out)) { return; }

    let act = select(0u, s_active[idx] & 1u, idx < arrayLength(&s_active));
    if (act != 0u) {
        plane_id_out[idx] = UINT_MAX;
        return;
    }
    var best = plane_id_in[idx];
    if (y > 0u) {
        let nidx = idx2(x, y - 1u, w);
        let nact = select(0u, s_active[nidx] & 1u, nidx < arrayLength(&s_active));
        if (nact == 0u && sameplane(idx, nidx) && plane_id_in[nidx] != UINT_MAX) {
            best = min(best, plane_id_in[nidx]);
        }
    }
    if (x + 1u < w) {
        let nidx = idx2(x + 1u, y, w);
        let nact = select(0u, s_active[nidx] & 1u, nidx < arrayLength(&s_active));
        if (nact == 0u && sameplane(idx, nidx) && plane_id_in[nidx] != UINT_MAX) {
            best = min(best, plane_id_in[nidx]);
        }
    }
    if (y + 1u < h) {
        let nidx = idx2(x, y + 1u, w);
        let nact = select(0u, s_active[nidx] & 1u, nidx < arrayLength(&s_active));
        if (nact == 0u && sameplane(idx, nidx) && plane_id_in[nidx] != UINT_MAX) {
            best = min(best, plane_id_in[nidx]);
        }
    }
    if (x > 0u) {
        let nidx = idx2(x - 1u, y, w);
        let nact = select(0u, s_active[nidx] & 1u, nidx < arrayLength(&s_active));
        if (nact == 0u && sameplane(idx, nidx) && plane_id_in[nidx] != UINT_MAX) {
            best = min(best, plane_id_in[nidx]);
        }
    }
    plane_id_out[idx] = best;
}
