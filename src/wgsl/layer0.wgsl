@group(0) @binding(0) var input_tex: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> input_img_info: array<u32>;

// 1-pass, 3 outputs (R/G/B)
@group(0) @binding(2) var<storage, read_write> output_r: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_g: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_b: array<u32>;

const EDGE_TH_R: i32 = 15;
const EDGE_TH_G: i32 = 15;
const EDGE_TH_B: i32 = 30;

fn load_rgb(x: i32, y: i32, width: u32, height: u32) -> vec3<i32> {
    if (width == 0u || height == 0u) {
        return vec3<i32>(0, 0, 0);
    }
    let cx = clamp(x, 0, i32(width) - 1);
    let cy = clamp(y, 0, i32(height) - 1);
    let p = textureLoad(input_tex, vec2<i32>(cx, cy), 0);
    return vec3<i32>(i32(p.x), i32(p.y), i32(p.z));
}

// 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW
fn dir_bit(dx: i32, dy: i32) -> u32 {
    if (dx == 0 && dy == -1) { return 1u << 0u; }
    if (dx == 1 && dy == -1) { return 1u << 1u; }
    if (dx == 1 && dy == 0)  { return 1u << 2u; }
    if (dx == 1 && dy == 1)  { return 1u << 3u; }
    if (dx == 0 && dy == 1)  { return 1u << 4u; }
    if (dx == -1 && dy == 1) { return 1u << 5u; }
    if (dx == -1 && dy == 0) { return 1u << 6u; }
    return 1u << 7u;
}

fn pack_channel(dir8: u32, val8: u32) -> u32 {
    // bit0: always 1 (store self regardless of active/inactive)
    // bit1..8: dir8
    // bit9..16: val8 (center channel value)
    return 1u | ((dir8 & 0xFFu) << 1u) | ((val8 & 0xFFu) << 9u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&input_img_info) < 2u) { return; }
    let height = input_img_info[0];
    let width  = input_img_info[1];
    if (width == 0u || height == 0u) { return; }

    let x: i32 = i32(gid.x);
    let y: i32 = i32(gid.y);
    if (x >= i32(width) || y >= i32(height)) { return; }

    let c: vec3<i32> = load_rgb(x, y, width, height);
    let rc: i32 = c.x;
    let gc: i32 = c.y;
    let bc: i32 = c.z;

    var dir8_r: u32 = 0u;
    var dir8_g: u32 = 0u;
    var dir8_b: u32 = 0u;

    // Channel-wise independent DIR8, but neighbor RGB loads are shared.
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }

            let n: vec3<i32> = load_rgb(x + dx, y + dy, width, height);
            let b: u32 = dir_bit(dx, dy);

            if (abs(rc - n.x) >= EDGE_TH_R) { dir8_r |= b; }
            if (abs(gc - n.y) >= EDGE_TH_G) { dir8_g |= b; }
            if (abs(bc - n.z) >= EDGE_TH_B) { dir8_b |= b; }
        }
    }

    let out_idx: u32 = u32(y) * width + u32(x);

    // Always store self: bit0 is forced to 1 for all pixels.
    let out_r_word: u32 = pack_channel(dir8_r, u32(rc) & 0xFFu);
    let out_g_word: u32 = pack_channel(dir8_g, u32(gc) & 0xFFu);
    let out_b_word: u32 = pack_channel(dir8_b, u32(bc) & 0xFFu);

    if (out_idx < arrayLength(&output_r)) { output_r[out_idx] = out_r_word; }
    if (out_idx < arrayLength(&output_g)) { output_g[out_idx] = out_g_word; }
    if (out_idx < arrayLength(&output_b)) { output_b[out_idx] = out_b_word; }
}
