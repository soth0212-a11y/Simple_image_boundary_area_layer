@group(0) @binding(0) var input_tex: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> input_img_info: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_masks: array<u32>;

const EDGE_TH: i32 = 10;

fn load_rgb(x: i32, y: i32, width: u32, height: u32) -> vec3<i32> {
    if (width == 0u || height == 0u) {
        return vec3<i32>(0, 0, 0);
    }
    let cx = clamp(x, 0, i32(width) - 1);
    let cy = clamp(y, 0, i32(height) - 1);
    let p = textureLoad(input_tex, vec2<i32>(cx, cy), 0);
    return vec3<i32>(i32(p.x), i32(p.y), i32(p.z));
}

fn load_rgb_unclamped(x: i32, y: i32) -> vec3<i32> {
    let p = textureLoad(input_tex, vec2<i32>(x, y), 0);
    return vec3<i32>(i32(p.x), i32(p.y), i32(p.z));
}

fn edge_hit_channel(center: i32, neighbor: i32) -> u32 {
    let d: i32 = abs(center - neighbor);
    return select(0u, 1u, d >= EDGE_TH);
}


@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tile_x: i32 = i32(gid.x);
    let tile_y: i32 = i32(gid.y);

    if (arrayLength(&input_img_info) < 2u) {
        return;
    }

    let height = input_img_info[0];
    let width  = input_img_info[1];
    if (width == 0u || height == 0u) {
        return;
    }

    let grid_w: i32 = i32(width);
    let grid_h: i32 = i32(height);
    if (tile_x >= grid_w || tile_y >= grid_h) {
        return;
    }

    var mask_r: u32 = 0u;
    var mask_g: u32 = 0u;
    var mask_b: u32 = 0u;
    let center: vec3<i32> = load_rgb(tile_x, tile_y, width, height);
    var dy: i32 = 0;
    loop {
        if (dy >= 3) { break; }
        var dx2: i32 = 0;
        loop {
            if (dx2 >= 3) { break; }
            if (dx2 != 1 || dy != 1) {
                let nx: i32 = tile_x + dx2 - 1;
                let ny: i32 = tile_y + dy - 1;
                let neighbor = load_rgb(nx, ny, width, height);
                var dir_mask: u32 = 0u;
                if (dx2 == 0 && dy == 0) {
                    dir_mask = (1u << 6u) | (1u << 7u) | (1u << 0u);
                } else if (dx2 == 1 && dy == 0) {
                    dir_mask = (1u << 7u) | (1u << 0u) | (1u << 1u);
                } else if (dx2 == 2 && dy == 0) {
                    dir_mask = (1u << 0u) | (1u << 1u) | (1u << 2u);
                } else if (dx2 == 2 && dy == 1) {
                    dir_mask = (1u << 1u) | (1u << 2u) | (1u << 3u);
                } else if (dx2 == 2 && dy == 2) {
                    dir_mask = (1u << 2u) | (1u << 3u) | (1u << 4u);
                } else if (dx2 == 1 && dy == 2) {
                    dir_mask = (1u << 3u) | (1u << 4u) | (1u << 5u);
                } else if (dx2 == 0 && dy == 2) {
                    dir_mask = (1u << 4u) | (1u << 5u) | (1u << 6u);
                } else if (dx2 == 0 && dy == 1) {
                    dir_mask = (1u << 5u) | (1u << 6u) | (1u << 7u);
                }
                let hit_r: u32 = edge_hit_channel(center.x, neighbor.x);
                let hit_g: u32 = edge_hit_channel(center.y, neighbor.y);
                let hit_b: u32 = edge_hit_channel(center.z, neighbor.z);
                if (hit_r == 1u) { mask_r = mask_r | dir_mask; }
                if (hit_g == 1u) { mask_g = mask_g | dir_mask; }
                if (hit_b == 1u) { mask_b = mask_b | dir_mask; }
            }
            dx2 = dx2 + 1;
        }
        dy = dy + 1;
    }

    let packed: u32 = (mask_r & 0xFFu) | ((mask_g & 0xFFu) << 8u) | ((mask_b & 0xFFu) << 16u);

    let out_idx: u32 = u32(tile_y * grid_w + tile_x);
    if (out_idx < arrayLength(&output_masks)) {
        output_masks[out_idx] = packed;
    }
}
