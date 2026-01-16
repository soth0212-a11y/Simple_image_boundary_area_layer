struct Params {
    w: u32,
    h: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> boundary4: array<u32>;
@group(0) @binding(1) var<storage, read> edge_label_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> edge_label_out: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

fn has_dir(mask: u32, bit: u32) -> bool {
    return (mask & bit) != 0u;
}

@compute @workgroup_size(16, 16, 1)
fn l2_label_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&boundary4) || idx >= arrayLength(&edge_label_out)) { return; }
    let m = boundary4[idx] & 0xFu;
    edge_label_out[idx] = select(UINT_MAX, idx, m != 0u);
}

@compute @workgroup_size(16, 16, 1)
fn l2_label_propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&edge_label_in) || idx >= arrayLength(&edge_label_out) || idx >= arrayLength(&boundary4)) {
        return;
    }

    let lab = edge_label_in[idx];
    if (lab == UINT_MAX) {
        edge_label_out[idx] = UINT_MAX;
        return;
    }
    let mask = boundary4[idx] & 0xFu;
    var best = lab;

    if (y > 0u) {
        let nidx = idx2(x, y - 1u, w);
        let nlab = edge_label_in[nidx];
        if (nlab != UINT_MAX) {
            let nmask = boundary4[nidx] & 0xFu;
            if (has_dir(mask, 1u) || has_dir(nmask, 4u)) {
                best = min(best, nlab);
            }
        }
    }
    if (x + 1u < w) {
        let nidx = idx2(x + 1u, y, w);
        let nlab = edge_label_in[nidx];
        if (nlab != UINT_MAX) {
            let nmask = boundary4[nidx] & 0xFu;
            if (has_dir(mask, 2u) || has_dir(nmask, 8u)) {
                best = min(best, nlab);
            }
        }
    }
    if (y + 1u < h) {
        let nidx = idx2(x, y + 1u, w);
        let nlab = edge_label_in[nidx];
        if (nlab != UINT_MAX) {
            let nmask = boundary4[nidx] & 0xFu;
            if (has_dir(mask, 4u) || has_dir(nmask, 1u)) {
                best = min(best, nlab);
            }
        }
    }
    if (x > 0u) {
        let nidx = idx2(x - 1u, y, w);
        let nlab = edge_label_in[nidx];
        if (nlab != UINT_MAX) {
            let nmask = boundary4[nidx] & 0xFu;
            if (has_dir(mask, 8u) || has_dir(nmask, 2u)) {
                best = min(best, nlab);
            }
        }
    }

    edge_label_out[idx] = best;
}
