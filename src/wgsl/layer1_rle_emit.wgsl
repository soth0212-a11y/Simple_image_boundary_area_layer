struct Params {
    w: u32,
    h: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Segment {
    x0: u32,
    x1: u32,
    y_color: u32,
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

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.w;
    let h = params.h;
    let y = gid.x;
    if (w == 0u || h == 0u) { return; }
    if (y >= h) { return; }
    if (y + 1u >= arrayLength(&row_offsets)) { return; }
    var write = row_offsets[y];
    let row_end = row_offsets[y + 1u];
    var in_seg: bool = false;
    var seg_x0: u32 = 0u;
    var seg_c: u32 = 0u;
    for (var x: u32 = 0u; x < w; x = x + 1u) {
        let idx = idx2(x, y, w);
        let a = select(0u, s_active[idx] & 1u, idx < arrayLength(&s_active));
        let c = select(0u, color565[idx] & 0xFFFFu, idx < arrayLength(&color565));
        if (a != 0u) {
            if (!in_seg) {
                in_seg = true;
                seg_x0 = x;
                seg_c = c;
            } else if (c != seg_c) {
                if (write < row_end && write < arrayLength(&segments)) {
                    segments[write] = Segment(seg_x0, x, (y << 16u) | (seg_c & 0xFFFFu), 0u);
                    write = write + 1u;
                }
                seg_x0 = x;
                seg_c = c;
            }
        } else if (in_seg) {
            if (write < row_end && write < arrayLength(&segments)) {
                segments[write] = Segment(seg_x0, x, (y << 16u) | (seg_c & 0xFFFFu), 0u);
                write = write + 1u;
            }
            in_seg = false;
        }
    }
    if (in_seg) {
        if (write < row_end && write < arrayLength(&segments)) {
            segments[write] = Segment(seg_x0, w, (y << 16u) | (seg_c & 0xFFFFu), 0u);
        }
    }
}
