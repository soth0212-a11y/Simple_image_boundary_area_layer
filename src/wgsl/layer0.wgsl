@group(0) @binding(0) var input_tex: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> input_img_info: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_cell_rgb: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_edge4: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_s_active: array<u32>;

const EDGE_TH_R: i32 = 15;
const EDGE_TH_G: i32 = 15;
const EDGE_TH_B: i32 = 20;

fn quantize_rgb565(c: vec3<i32>) -> vec3<i32> {
    let r = (c.x >> 3) & 31;
    let g = (c.y >> 2) & 63;
    let b = (c.z >> 3) & 31;
    return vec3<i32>((r << 3), (g << 2), (b << 3));
}

fn pack_rgb565(r: u32, g: u32, b: u32) -> u32 {
    let r5 = (r >> 3) & 31u;
    let g6 = (g >> 2) & 63u;
    let b5 = (b >> 3) & 31u;
    return (r5 << 11u) | (g6 << 5u) | b5;
}

fn load_rgb(x: i32, y: i32, width: u32, height: u32) -> vec3<i32> {
    if (width == 0u || height == 0u) {
        return vec3<i32>(0, 0, 0);
    }
    let cx = clamp(x, 0, i32(width) - 1);
    let cy = clamp(y, 0, i32(height) - 1);
    let p = textureLoad(input_tex, vec2<i32>(cx, cy), 0);
    let c = vec3<i32>(i32(p.x), i32(p.y), i32(p.z));
    return quantize_rgb565(c);
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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (arrayLength(&input_img_info) < 2u) { return; }
    let height = input_img_info[0];
    let width  = input_img_info[1];
    if (width == 0u || height == 0u) { return; }
    var edge_th_r: i32 = EDGE_TH_R;
    var edge_th_g: i32 = EDGE_TH_G;
    var edge_th_b: i32 = EDGE_TH_B;
    var min_channels: u32 = 3u;
    var pixel_min_dirs: u32 = 1u;
    if (arrayLength(&input_img_info) >= 7u) {
        edge_th_r = i32(input_img_info[2]);
        edge_th_g = i32(input_img_info[3]);
        edge_th_b = i32(input_img_info[4]);
        let v = input_img_info[5];
        min_channels = clamp(v, 1u, 3u);
        let p = input_img_info[6];
        pixel_min_dirs = clamp(p, 1u, 8u);
    }

    let out_w: u32 = width;
    let out_h: u32 = height;
    let ox: u32 = gid.x;
    let oy: u32 = gid.y;
    if (ox >= out_w || oy >= out_h) { return; }

    let base_x: u32 = ox;
    let base_y: u32 = oy;
    var active_cnt: u32 = 0u;
    var inactive_cnt: u32 = 0u;
    var sum_r_act: u32 = 0u;
    var sum_g_act: u32 = 0u;
    var sum_b_act: u32 = 0u;
    var sum_r_in: u32 = 0u;
    var sum_g_in: u32 = 0u;
    var sum_b_in: u32 = 0u;
    var dir_or: u32 = 0u;

    for (var dy: u32 = 0u; dy < 1u; dy = dy + 1u) {
        let y: u32 = base_y + dy;
        if (y >= height) { continue; }
        for (var dx: u32 = 0u; dx < 1u; dx = dx + 1u) {
            let x: u32 = base_x + dx;
            if (x >= width) { continue; }
            let c: vec3<i32> = load_rgb(i32(x), i32(y), width, height);
            let rc: i32 = c.x;
            let gc: i32 = c.y;
            let bc: i32 = c.z;
            var dir8_r: u32 = 0u;
            var dir8_g: u32 = 0u;
            var dir8_b: u32 = 0u;
            for (var ddy: i32 = -1; ddy <= 1; ddy = ddy + 1) {
                for (var ddx: i32 = -1; ddx <= 1; ddx = ddx + 1) {
                    if (ddx == 0 && ddy == 0) { continue; }
                    let n: vec3<i32> = load_rgb(i32(x) + ddx, i32(y) + ddy, width, height);
                    let b: u32 = dir_bit(ddx, ddy);
                    if (abs(rc - n.x) >= edge_th_r) { dir8_r |= b; }
                    if (abs(gc - n.y) >= edge_th_g) { dir8_g |= b; }
                    if (abs(bc - n.z) >= edge_th_b) { dir8_b |= b; }
                }
            }
            var dir8: u32 = 0u;
            for (var bit: u32 = 0u; bit < 8u; bit = bit + 1u) {
                let mask: u32 = 1u << bit;
                let cnt: u32 = select(0u, 1u, (dir8_r & mask) != 0u) +
                               select(0u, 1u, (dir8_g & mask) != 0u) +
                               select(0u, 1u, (dir8_b & mask) != 0u);
                if (cnt >= min_channels) { dir8 = dir8 | mask; }
            }
            var dir_cnt: u32 = 0u;
            for (var bit2: u32 = 0u; bit2 < 8u; bit2 = bit2 + 1u) {
                dir_cnt = dir_cnt + select(0u, 1u, (dir8 & (1u << bit2)) != 0u);
            }
            let pixel_active: bool = dir_cnt >= pixel_min_dirs;
            let rv: u32 = u32(rc) & 0xFFu;
            let gv: u32 = u32(gc) & 0xFFu;
            let bv: u32 = u32(bc) & 0xFFu;
            if (pixel_active) {
                active_cnt = active_cnt + 1u;
                sum_r_act = sum_r_act + rv;
                sum_g_act = sum_g_act + gv;
                sum_b_act = sum_b_act + bv;
                dir_or = dir_or | dir8;
            } else {
                inactive_cnt = inactive_cnt + 1u;
                sum_r_in = sum_r_in + rv;
                sum_g_in = sum_g_in + gv;
                sum_b_in = sum_b_in + bv;
            }
        }
    }

    let block_active: bool = active_cnt >= inactive_cnt;
    let sel_cnt: u32 = select(inactive_cnt, active_cnt, block_active);
    let denom: u32 = max(sel_cnt, 1u);
    let avg_r: u32 = select(sum_r_in, sum_r_act, block_active) / denom;
    let avg_g: u32 = select(sum_g_in, sum_g_act, block_active) / denom;
    let avg_b: u32 = select(sum_b_in, sum_b_act, block_active) / denom;
    let dir8: u32 = select(0u, dir_or, block_active) & 0xFFu;
    let q565: u32 = pack_rgb565(avg_r, avg_g, avg_b);
    let out0: u32 = select(0u, 1u, block_active) | (dir8 << 1u) | (q565 << 9u);
    let out1: u32 = (avg_r & 0xFFu) | ((avg_g & 0xFFu) << 8u) | ((avg_b & 0xFFu) << 16u);
    let edge4: u32 = ((dir8 >> 0u) & 1u) |
                     (((dir8 >> 2u) & 1u) << 1u) |
                     (((dir8 >> 4u) & 1u) << 2u) |
                     (((dir8 >> 6u) & 1u) << 3u);

    let out_idx: u32 = oy * out_w + ox;
    let base: u32 = out_idx * 2u;
    if (base + 1u < arrayLength(&output_packed)) {
        output_packed[base] = out0;
        output_packed[base + 1u] = out1;
    }
    if (out_idx < arrayLength(&output_cell_rgb)) {
        output_cell_rgb[out_idx] = out1;
    }
    if (out_idx < arrayLength(&output_edge4)) {
        output_edge4[out_idx] = edge4;
    }
    if (out_idx < arrayLength(&output_s_active)) {
        output_s_active[out_idx] = out0 & 1u;
    }
}
