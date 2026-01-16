struct Params {
    w: u32,
    h: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> plane_id: array<u32>;
@group(0) @binding(1) var<storage, read> edge4_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> boundary4: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const UINT_MAX: u32 = 0xFFFFFFFFu;

fn idx2(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

@compute @workgroup_size(16, 16, 1)
fn l2_boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    if (w == 0u || h == 0u) { return; }
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) { return; }
    let idx = idx2(x, y, w);
    if (idx >= arrayLength(&boundary4) || idx >= arrayLength(&plane_id) || idx >= arrayLength(&edge4_in)) {
        return;
    }
    let pid0 = plane_id[idx];
    if (pid0 == UINT_MAX) {
        boundary4[idx] = 0u;
        return;
    }
    let edge = edge4_in[idx] & 0xFu;
    if (edge == 0u) {
        boundary4[idx] = 0u;
        return;
    }

    var out_mask: u32 = 0u;

    if ((edge & 1u) != 0u) { // N
        if (y == 0u) {
            out_mask = out_mask | 1u;
        } else {
            let nidx = idx2(x, y - 1u, w);
            let pid1 = select(UINT_MAX, plane_id[nidx], nidx < arrayLength(&plane_id));
            if (pid1 == UINT_MAX || pid1 != pid0) { out_mask = out_mask | 1u; }
        }
    }
    if ((edge & 2u) != 0u) { // E
        if (x + 1u >= w) {
            out_mask = out_mask | 2u;
        } else {
            let nidx = idx2(x + 1u, y, w);
            let pid1 = select(UINT_MAX, plane_id[nidx], nidx < arrayLength(&plane_id));
            if (pid1 == UINT_MAX || pid1 != pid0) { out_mask = out_mask | 2u; }
        }
    }
    if ((edge & 4u) != 0u) { // S
        if (y + 1u >= h) {
            out_mask = out_mask | 4u;
        } else {
            let nidx = idx2(x, y + 1u, w);
            let pid1 = select(UINT_MAX, plane_id[nidx], nidx < arrayLength(&plane_id));
            if (pid1 == UINT_MAX || pid1 != pid0) { out_mask = out_mask | 4u; }
        }
    }
    if ((edge & 8u) != 0u) { // W
        if (x == 0u) {
            out_mask = out_mask | 8u;
        } else {
            let nidx = idx2(x - 1u, y, w);
            let pid1 = select(UINT_MAX, plane_id[nidx], nidx < arrayLength(&plane_id));
            if (pid1 == UINT_MAX || pid1 != pid0) { out_mask = out_mask | 8u; }
        }
    }

    boundary4[idx] = out_mask & 0xFu;
}
