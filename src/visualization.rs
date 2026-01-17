use image::{Rgba, RgbaImage};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SAVE_DIR: &str = "result_images";

#[inline]
pub fn span_1d(cell: usize, grid: usize, orig: usize) -> (usize, usize) {
    if grid == 0 || orig == 0 {
        return (0, 0);
    }
    let s0 = (cell * orig) / grid;
    let s1 = (((cell + 1) * orig) / grid).saturating_sub(1);
    (s0.min(orig - 1), s1.min(orig - 1))
}

fn high_vis_dir_colors() -> [Rgba<u8>; 8] {
    [
        Rgba([0, 255, 255, 255]),   // N  cyan
        Rgba([255, 255, 0, 255]),   // NE yellow
        Rgba([255, 0, 255, 255]),   // E  magenta
        Rgba([255, 128, 0, 255]),   // SE orange
        Rgba([255, 0, 0, 255]),     // S  red
        Rgba([0, 255, 0, 255]),     // SW lime
        Rgba([0, 128, 255, 255]),   // W  sky
        Rgba([255, 255, 255, 255]), // NW white
    ]
}

fn l3_palette() -> [Rgba<u8>; 8] {
    [
        Rgba([255, 64, 64, 255]),
        Rgba([64, 255, 64, 255]),
        Rgba([64, 64, 255, 255]),
        Rgba([255, 255, 64, 255]),
        Rgba([255, 64, 255, 255]),
        Rgba([64, 255, 255, 255]),
        Rgba([255, 160, 64, 255]),
        Rgba([255, 255, 255, 255]),
    ]
}

#[inline]
fn draw_rect_outline(
    img: &mut RgbaImage,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color: Rgba<u8>,
) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return;
    }
    let sx0 = x0.min(w - 1);
    let sy0 = y0.min(h - 1);
    let sx1 = x1.min(w - 1);
    let sy1 = y1.min(h - 1);
    if sx0 > sx1 || sy0 > sy1 {
        return;
    }
    for x in sx0..=sx1 {
        img.put_pixel(x as u32, sy0 as u32, color);
        img.put_pixel(x as u32, sy1 as u32, color);
    }
    for y in sy0..=sy1 {
        img.put_pixel(sx0 as u32, y as u32, color);
        img.put_pixel(sx1 as u32, y as u32, color);
    }
}

#[inline]
fn draw_rect_outline_alt(
    img: &mut RgbaImage,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color_a: Rgba<u8>,
    color_b: Rgba<u8>,
) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return;
    }
    let sx0 = x0.min(w - 1);
    let sy0 = y0.min(h - 1);
    let sx1 = x1.min(w - 1);
    let sy1 = y1.min(h - 1);
    if sx0 > sx1 || sy0 > sy1 {
        return;
    }
    for x in sx0..=sx1 {
        let top = if (x.wrapping_add(sy0) & 1) == 0 { color_a } else { color_b };
        let bottom = if (x.wrapping_add(sy1) & 1) == 0 { color_a } else { color_b };
        img.put_pixel(x as u32, sy0 as u32, top);
        img.put_pixel(x as u32, sy1 as u32, bottom);
    }
    for y in sy0..=sy1 {
        let left = if (sx0.wrapping_add(y) & 1) == 0 { color_a } else { color_b };
        let right = if (sx1.wrapping_add(y) & 1) == 0 { color_a } else { color_b };
        img.put_pixel(sx0 as u32, y as u32, left);
        img.put_pixel(sx1 as u32, y as u32, right);
    }
}

#[inline]
fn fill_rect(img: &mut RgbaImage, x0: usize, y0: usize, x1: usize, y1: usize, color: Rgba<u8>) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return;
    }
    let sx0 = x0.min(w - 1);
    let sy0 = y0.min(h - 1);
    let sx1 = x1.min(w - 1);
    let sy1 = y1.min(h - 1);
    for y in sy0..=sy1 {
        for x in sx0..=sx1 {
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}

#[inline]
fn draw_dir_band(
    img: &mut RgbaImage,
    sx0: usize,
    sy0: usize,
    sx1: usize,
    sy1: usize,
    thickness: usize,
    dir: u32,
    color: Rgba<u8>,
) {
    if thickness == 0 {
        return;
    }
    let tx = (sx0 + thickness - 1).min(sx1);
    let ty = (sy0 + thickness - 1).min(sy1);
    let bx = sx1.saturating_sub(thickness - 1);
    let by = sy1.saturating_sub(thickness - 1);
    match dir {
        0 => fill_rect(img, sx0, sy0, sx1, ty, color), // N
        2 => fill_rect(img, bx, sy0, sx1, sy1, color), // E
        4 => fill_rect(img, sx0, by, sx1, sy1, color), // S
        6 => fill_rect(img, sx0, sy0, tx, sy1, color), // W
        _ => {}
    }
}

pub fn save_layer0_dir_overlay(
    src_path: &Path,
    layer0: &[u32],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 || layer0.len() < grid_w * grid_h {
        return Ok(());
    }
    let base_img = image::open(src_path)?.to_rgba8();
    let img_w = base_img.width() as usize;
    let img_h = base_img.height() as usize;
    if img_w == 0 || img_h == 0 {
        return Ok(());
    }

    let mut img_r = base_img.clone();
    let mut img_g = base_img.clone();
    let mut img_b = base_img;
    let colors = high_vis_dir_colors();
    let band: usize = 1;

    for y in 0..grid_h {
        for x in 0..grid_w {
            let idx = y * grid_w + x;
            let mask_r = layer0[idx] & 0xFFu32;
            let mask_g = (layer0[idx] >> 8u32) & 0xFFu32;
            let mask_b = (layer0[idx] >> 16u32) & 0xFFu32;
            let (sx0, sx1) = span_1d(x, grid_w, img_w);
            let (sy0, sy1) = span_1d(y, grid_h, img_h);
            for bit in 0..8u32 {
                let color = colors[bit as usize];
                if (mask_r & (1u32 << bit)) != 0u32 {
                    match bit {
                        0 | 2 | 4 | 6 => draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, bit, color),
                        1 => { draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, 0, color); draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, 2, color); }
                        3 => { draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, 2, color); draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, 4, color); }
                        5 => { draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, 4, color); draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, 6, color); }
                        7 => { draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, 0, color); draw_dir_band(&mut img_r, sx0, sy0, sx1, sy1, band, 6, color); }
                        _ => {}
                    }
                }
                if (mask_g & (1u32 << bit)) != 0u32 {
                    match bit {
                        0 | 2 | 4 | 6 => draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, bit, color),
                        1 => { draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, 0, color); draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, 2, color); }
                        3 => { draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, 2, color); draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, 4, color); }
                        5 => { draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, 4, color); draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, 6, color); }
                        7 => { draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, 0, color); draw_dir_band(&mut img_g, sx0, sy0, sx1, sy1, band, 6, color); }
                        _ => {}
                    }
                }
                if (mask_b & (1u32 << bit)) != 0u32 {
                    match bit {
                        0 | 2 | 4 | 6 => draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, bit, color),
                        1 => { draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, 0, color); draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, 2, color); }
                        3 => { draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, 2, color); draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, 4, color); }
                        5 => { draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, 4, color); draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, 6, color); }
                        7 => { draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, 0, color); draw_dir_band(&mut img_b, sx0, sy0, sx1, sy1, band, 6, color); }
                        _ => {}
                    }
                }
            }
        }
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img_r.save(Path::new(SAVE_DIR).join(format!("{}_layer0_r_{}_{}.png", stem, name_suffix, time)))?;
    img_g.save(Path::new(SAVE_DIR).join(format!("{}_layer0_g_{}_{}.png", stem, name_suffix, time)))?;
    img_b.save(Path::new(SAVE_DIR).join(format!("{}_layer0_b_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

pub fn save_layer0_rgb_overlay(
    src_path: &Path,
    r: &[u32],
    g: &[u32],
    b: &[u32],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 {
        return Ok(());
    }
    let count = grid_w * grid_h;
    if r.len() < count || g.len() < count || b.len() < count {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(grid_w as u32, grid_h as u32);
    for y in 0..grid_h {
        for x in 0..grid_w {
            let idx = y * grid_w + x;
            let mr = (r[idx] >> 1) & 0xFF;
            let mg = (g[idx] >> 1) & 0xFF;
            let mb = (b[idx] >> 1) & 0xFF;
            let vr = (mr.count_ones() * 32).min(255) as u8;
            let vg = (mg.count_ones() * 32).min(255) as u8;
            let vb = (mb.count_ones() * 32).min(255) as u8;
            img.put_pixel(x as u32, y as u32, Rgba([vr, vg, vb, 255]));
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer0_rgb_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

pub fn save_layer0_packed_rgb(
    src_path: &Path,
    packed: &[u32],
    out_w: usize,
    out_h: usize,
    name_suffix: &str,
) -> Result<(), image::ImageError> {
    if out_w == 0 || out_h == 0 {
        return Ok(());
    }
    let count = out_w * out_h * 2;
    if packed.len() < count {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(out_w as u32, out_h as u32);
    for y in 0..out_h {
        for x in 0..out_w {
            let idx = (y * out_w + x) * 2;
            let flags = packed[idx];
            let is_active = (flags & 1u32) != 0u32;
            let color = if is_active {
                Rgba([200, 200, 200, 255])
            } else {
                let q565 = (flags >> 9) & 0xFFFF;
                let r5 = (q565 >> 11) & 31;
                let g6 = (q565 >> 5) & 63;
                let b5 = q565 & 31;
                let r = ((r5 << 3) | (r5 >> 2)) as u8;
                let g = ((g6 << 2) | (g6 >> 4)) as u8;
                let b = ((b5 << 3) | (b5 >> 2)) as u8;
                Rgba([r, g, b, 255])
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer0_rgb_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

fn color565_to_rgb(c: u32) -> Rgba<u8> {
    let q565 = c & 0xFFFFu32;
    let r5 = (q565 >> 11) & 31;
    let g6 = (q565 >> 5) & 63;
    let b5 = q565 & 31;
    let r = ((r5 << 3) | (r5 >> 2)) as u8;
    let g = ((g6 << 2) | (g6 >> 4)) as u8;
    let b = ((b5 << 3) | (b5 >> 2)) as u8;
    Rgba([r, g, b, 255])
}

pub fn save_l1_segments_overlay(
    src_path: &Path,
    segments: &[crate::rle_ccl_gpu::Segment],
    color565_map: &[u32],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(grid_w as u32, grid_h as u32);
    for seg in segments {
        let x0 = (seg.tl & 0xFFFFu32) as usize;
        let y0 = (seg.tl >> 16) as usize;
        let x1 = (seg.br & 0xFFFFu32) as usize;
        let y1 = (seg.br >> 16) as usize;
        if x1 == 0 || y1 == 0 || x0 >= grid_w || y0 >= grid_h {
            continue;
        }
        let x1i = x1.saturating_sub(1).min(grid_w.saturating_sub(1));
        let y1i = y1.saturating_sub(1).min(grid_h.saturating_sub(1));
        let cx = (x0 + x1i) / 2;
        let cy = (y0 + y1i) / 2;
        let idx = cy.saturating_mul(grid_w).saturating_add(cx);
        let c = if idx < color565_map.len() {
            color565_map[idx] & 0xFFFFu32
        } else {
            seg.color565 & 0xFFFFu32
        };
        let color = color565_to_rgb(c);
        draw_rect_outline(&mut img, x0.min(grid_w - 1), y0.min(grid_h - 1), x1i, y1i, color);
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l1_segments_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

pub fn save_l2_boxes_on_src(
    src_path: &Path,
    boxes: &[crate::rle_ccl_gpu::OutBox],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 {
        return Ok(());
    }
    let mut img = image::open(src_path)?.to_rgba8();
    let img_w = img.width() as usize;
    let img_h = img.height() as usize;
    if img_w == 0 || img_h == 0 {
        return Ok(());
    }
    for b in boxes {
        let x0 = (b.x0y0 & 0xFFFFu32) as usize;
        let y0 = (b.x0y0 >> 16) as usize;
        let x1 = (b.x1y1 & 0xFFFFu32) as usize;
        let y1 = (b.x1y1 >> 16) as usize;
        if x1 == 0 || y1 == 0 {
            continue;
        }
        let x1i = x1.saturating_sub(1);
        let y1i = y1.saturating_sub(1);
        let (sx0a, sx0b) = span_1d(x0, grid_w, img_w);
        let (sx1a, sx1b) = span_1d(x1i, grid_w, img_w);
        let (sy0a, sy0b) = span_1d(y0, grid_h, img_h);
        let (sy1a, sy1b) = span_1d(y1i, grid_h, img_h);
        let min_x = sx0a.min(sx0b).min(sx1a).min(sx1b);
        let max_x = sx0a.max(sx0b).max(sx1a).max(sx1b);
        let min_y = sy0a.min(sy0b).min(sy1a).min(sy1b);
        let max_y = sy0a.max(sy0b).max(sy1a).max(sy1b);
        let red = Rgba([255, 0, 0, 255]);
        draw_rect_outline(&mut img, min_x, min_y, max_x, max_y, red);
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l2_boxes_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

pub fn save_l3_boxes_on_src(
    src_path: &Path,
    boxes: &[crate::rle_ccl_gpu::OutBox],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 {
        return Ok(());
    }
    let mut img = image::open(src_path)?.to_rgba8();
    let img_w = img.width() as usize;
    let img_h = img.height() as usize;
    if img_w == 0 || img_h == 0 {
        return Ok(());
    }
    let palette = l3_palette();
    for (idx, b) in boxes.iter().enumerate() {
        let x0 = (b.x0y0 & 0xFFFFu32) as usize;
        let y0 = (b.x0y0 >> 16) as usize;
        let x1 = (b.x1y1 & 0xFFFFu32) as usize;
        let y1 = (b.x1y1 >> 16) as usize;
        if x1 == 0 || y1 == 0 {
            continue;
        }
        let x1i = x1.saturating_sub(1);
        let y1i = y1.saturating_sub(1);
        let (sx0a, sx0b) = span_1d(x0, grid_w, img_w);
        let (sx1a, sx1b) = span_1d(x1i, grid_w, img_w);
        let (sy0a, sy0b) = span_1d(y0, grid_h, img_h);
        let (sy1a, sy1b) = span_1d(y1i, grid_h, img_h);
        let min_x = sx0a.min(sx0b).min(sx1a).min(sx1b);
        let max_x = sx0a.max(sx0b).max(sx1a).max(sx1b);
        let min_y = sy0a.min(sy0b).min(sy1a).min(sy1b);
        let max_y = sy0a.max(sy0b).max(sy1a).max(sy1b);
        let color = palette[idx % palette.len()];
        draw_rect_outline(&mut img, min_x, min_y, max_x, max_y, color);
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l3_boxes_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

pub fn save_l1_bbox_on_src(
    src_path: &Path,
    bbox0: &[u32],
    bbox1: &[u32],
    bbox_color: &[u32],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 {
        return Ok(());
    }
    if bbox0.len() < grid_w * grid_h || bbox1.len() < grid_w * grid_h || bbox_color.len() < grid_w * grid_h {
        return Ok(());
    }
    let mut img = image::open(src_path)?.to_rgba8();
    let img_w = img.width() as usize;
    let img_h = img.height() as usize;
    if img_w == 0 || img_h == 0 {
        return Ok(());
    }
    for y in 0..grid_h {
        for x in 0..grid_w {
            let idx = y * grid_w + x;
            if bbox0[idx] == 0xFFFFFFFFu32 || bbox1[idx] == 0xFFFFFFFFu32 {
                continue;
            }
            let x0 = (bbox0[idx] & 0xFFFFu32) as usize;
            let y0 = (bbox0[idx] >> 16) as usize;
            let x1 = (bbox1[idx] & 0xFFFFu32) as usize;
            let y1 = (bbox1[idx] >> 16) as usize;
            if x1 == 0 || y1 == 0 {
                continue;
            }
            let x1i = x1.saturating_sub(1);
            let y1i = y1.saturating_sub(1);
            let (sx0a, sx0b) = span_1d(x0, grid_w, img_w);
            let (sx1a, sx1b) = span_1d(x1i, grid_w, img_w);
            let (sy0a, sy0b) = span_1d(y0, grid_h, img_h);
            let (sy1a, sy1b) = span_1d(y1i, grid_h, img_h);
            let min_x = sx0a.min(sx0b).min(sx1a).min(sx1b);
            let max_x = sx0a.max(sx0b).max(sx1a).max(sx1b);
            let min_y = sy0a.min(sy0b).min(sy1a).min(sy1b);
            let max_y = sy0a.max(sy0b).max(sy1a).max(sy1b);
            let q565 = bbox_color[idx] & 0xFFFFu32;
            let r5 = (q565 >> 11) & 31;
            let g6 = (q565 >> 5) & 63;
            let b5 = q565 & 31;
            let r = ((r5 << 3) | (r5 >> 2)) as u8;
            let g = ((g6 << 2) | (g6 >> 4)) as u8;
            let b = ((b5 << 3) | (b5 >> 2)) as u8;
            let color = Rgba([r, g, b, 255]);
            let red = Rgba([255, 0, 0, 255]);
            draw_rect_outline_alt(&mut img, min_x, min_y, max_x, max_y, red, color);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l1_bbox_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

pub fn save_layer0_channel_overlay(
    src_path: &Path,
    channel: &[u32],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
    channel_tag: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 || channel.len() < grid_w * grid_h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(grid_w as u32, grid_h as u32);
    for y in 0..grid_h {
        for x in 0..grid_w {
            let idx = y * grid_w + x;
            let dir8 = (channel[idx] >> 1) & 0xFF;
            let val = ((channel[idx] >> 9) & 0xFF) as u8;
            let color = if dir8 != 0 {
                Rgba([200, 200, 200, 255])
            } else {
                Rgba([val, val, val, 255])
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer0_{}_{}_{}.png", stem, channel_tag, name_suffix, time)))?;
    Ok(())
}

pub fn log_timing_layers(
    src_path: &Path,
    img_info: [u32; 4],
    l0: std::time::Duration,
    l1: std::time::Duration,
    l2: std::time::Duration,
    total: std::time::Duration,
) {
    static LOG_WRITER: OnceLock<Mutex<BufWriter<std::fs::File>>> = OnceLock::new();
    let writer = LOG_WRITER.get_or_init(|| {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let _ = fs::create_dir_all("logs");
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("logs/layer_timing_{}.log", ts))
            .unwrap();
        Mutex::new(BufWriter::new(file))
    });
    if let Ok(mut w) = writer.lock() {
        let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        writeln!(w, "image: {} ({}x{})", stem, img_info[1], img_info[0]).ok();
        writeln!(w, "layer0: {:?}", l0).ok();
        writeln!(w, "layer1: {:?}", l1).ok();
        writeln!(w, "layer2: {:?}", l2).ok();
        writeln!(w, "total: {:?}\n", total).ok();
        let _ = w.flush();
    }
}

pub fn log_timing_layers_l2_passes(
    src_path: &Path,
    img_info: [u32; 4],
    l0: std::time::Duration,
    l1: std::time::Duration,
    l2: std::time::Duration,
    total: std::time::Duration,
    l2_passes: &[(&'static str, std::time::Duration)],
) {
    static LOG_WRITER: OnceLock<Mutex<BufWriter<std::fs::File>>> = OnceLock::new();
    let writer = LOG_WRITER.get_or_init(|| {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let _ = fs::create_dir_all("logs");
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("logs/layer_timing_{}.log", ts))
            .unwrap();
        Mutex::new(BufWriter::new(file))
    });
    if let Ok(mut w) = writer.lock() {
        let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        writeln!(w, "image: {} ({}x{})", stem, img_info[1], img_info[0]).ok();
        writeln!(w, "layer0: {:?}", l0).ok();
        writeln!(w, "layer1: {:?}", l1).ok();
        writeln!(w, "layer2: {:?}", l2).ok();
        for (name, dur) in l2_passes {
            writeln!(w, "  {}: {:?}", name, dur).ok();
        }
        writeln!(w, "total: {:?}\n", total).ok();
        let _ = w.flush();
    }
}
