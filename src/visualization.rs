use image::{Rgba, RgbaImage};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

// model.rs에서 정의된 타입을 사용합니다.
pub const SAVE_DIR: &str = "result_images";

// ======= 보조 함수 (밀림 방지 매핑) =======

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
fn blend_pixel(dst: &mut Rgba<u8>, src: Rgba<u8>, alpha: u8) {
    let a = alpha as u16;
    let inv = 255u16 - a;
    dst.0[0] = (((dst.0[0] as u16 * inv) + (src.0[0] as u16 * a)) / 255) as u8;
    dst.0[1] = (((dst.0[1] as u16 * inv) + (src.0[1] as u16 * a)) / 255) as u8;
    dst.0[2] = (((dst.0[2] as u16 * inv) + (src.0[2] as u16 * a)) / 255) as u8;
}

fn blend_rect(img: &mut RgbaImage, x0: usize, y0: usize, x1: usize, y1: usize, color: Rgba<u8>, alpha: u8) {
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
            let mut px = *img.get_pixel(x as u32, y as u32);
            blend_pixel(&mut px, color, alpha);
            img.put_pixel(x as u32, y as u32, px);
        }
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

fn l1_boundary_mask(plane_id: &[u32], w: usize, h: usize) -> Vec<u8> {
    let mut mask = vec![0u8; w * h];
    if w == 0 || h == 0 || plane_id.len() < w * h {
        return mask;
    }
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let v = plane_id[idx];
            let mut m = 0u8;
            if y > 0 {
                let n = plane_id[(y - 1) * w + x];
                if n != v { m |= 1u8; }
            }
            if x + 1 < w {
                let n = plane_id[y * w + (x + 1)];
                if n != v { m |= 2u8; }
            }
            if y + 1 < h {
                let n = plane_id[(y + 1) * w + x];
                if n != v { m |= 4u8; }
            }
            if x > 0 {
                let n = plane_id[y * w + (x - 1)];
                if n != v { m |= 8u8; }
            }
            mask[idx] = m;
        }
    }
    mask
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
            let rgb = packed[idx + 1];
            let is_active = (flags & 1u32) != 0u32;
            let color = if is_active {
                Rgba([200, 200, 200, 255])
            } else {
                let r = (rgb & 0xFF) as u8;
                let g = ((rgb >> 8) & 0xFF) as u8;
                let b = ((rgb >> 16) & 0xFF) as u8;
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
                match channel_tag {
                    "r" => Rgba([val, 0, 0, 255]),
                    "g" => Rgba([0, val, 0, 255]),
                    "b" => Rgba([0, 0, val, 255]),
                    _ => Rgba([val, val, val, 255]),
                }
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



pub fn save_layer1_mask_overlay(
    src_path: &Path,
    masks: &[u32],
    grid3_w: usize,
    grid3_h: usize,
    name_suffix: &str,
) -> Result<(), image::ImageError> {
    if grid3_w == 0 || grid3_h == 0 {
        return Ok(());
    }
    let cell_count = grid3_w * grid3_h;
    if masks.len() < cell_count {
        return Ok(());
    }

    let mut img = image::open(src_path)?.to_rgba8();
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    if orig_w == 0 || orig_h == 0 {
        return Ok(());
    }

    let color_active = Rgba([200, 200, 200, 255]);
    let color_inactive = Rgba([128, 128, 128, 255]);

    for idx in 0..cell_count {
        let m = masks[idx];
        let active = (m >> 16) & 0xFFFF;
        let total = m & 0xFFFF;
        let is_active = total != 0 && active * 2 >= total;

        let x = idx % grid3_w;
        let y = idx / grid3_w;
        let (sx0, sx1) = span_1d(x, grid3_w, orig_w);
        let (sy0, sy1) = span_1d(y, grid3_h, orig_h);
        let color = if is_active { color_active } else { color_inactive };
        fill_rect(&mut img, sx0, sy0, sx1, sy1, color);
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer1_mask_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

pub fn save_layer1_channel(
    src_path: &Path,
    data: &[u32],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
    channel: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 || data.len() < grid_w * grid_h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(grid_w as u32, grid_h as u32);
    for y in 0..grid_h {
        for x in 0..grid_w {
            let idx = y * grid_w + x;
            let v = (data[idx] & 0xFFu32) as u8;
            let color = match channel {
                "r" => Rgba([v, 0, 0, 255]),
                "g" => Rgba([0, v, 0, 255]),
                "b" => Rgba([0, 0, v, 255]),
                _ => Rgba([v, v, v, 255]),
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer1_{}_{}_{}.png", stem, channel, name_suffix, time)))?;
    Ok(())
}

pub fn save_layer1_channel_with_mask(
    src_path: &Path,
    mask: &[u32],
    data: &[u32],
    grid_w: usize,
    grid_h: usize,
    name_suffix: &str,
    channel: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 {
        return Ok(());
    }
    let count = grid_w * grid_h;
    if data.len() < count || mask.len() < count {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(grid_w as u32, grid_h as u32);
    for y in 0..grid_h {
        for x in 0..grid_w {
            let idx = y * grid_w + x;
            let m = mask[idx];
            let active = (m >> 16) & 0xFFFF;
            let total = m & 0xFFFF;
            let v = (data[idx] & 0xFFu32) as u8;
            let color = if total != 0 && active == total {
                Rgba([200, 200, 200, 255])
            } else {
                match channel {
                    "r" => Rgba([v, 0, 0, 255]),
                    "g" => Rgba([0, v, 0, 255]),
                    "b" => Rgba([0, 0, v, 255]),
                    _ => Rgba([v, v, v, 255]),
                }
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer1_{}_{}_{}.png", stem, channel, name_suffix, time)))?;
    Ok(())
}

pub fn save_layer1_rgb_composite(
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
            let mut active_channels: u32 = 0u32;
            if ((r[idx] >> 8) & 0xFFu32) != 0u32 { active_channels = active_channels + 1u32; }
            if ((g[idx] >> 8) & 0xFFu32) != 0u32 { active_channels = active_channels + 1u32; }
            if ((b[idx] >> 8) & 0xFFu32) != 0u32 { active_channels = active_channels + 1u32; }
            let color = if active_channels >= 2u32 {
                Rgba([200, 200, 200, 255])
            } else {
                let rv = (r[idx] & 0xFFu32) as u8;
                let gv = (g[idx] & 0xFFu32) as u8;
                let bv = (b[idx] & 0xFFu32) as u8;
                Rgba([rv, gv, bv, 255])
            };
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer1_rgb_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

fn l1_mask_to_rgb(mask: u32) -> Rgba<u8> {
    let mut r = 0u8;
    let mut g = 0u8;
    let mut b = 0u8;
    if (mask & 1u32) != 0u32 { r = 255; }
    if (mask & 2u32) != 0u32 { g = 255; }
    if (mask & 4u32) != 0u32 { b = 255; }
    if (mask & 8u32) != 0u32 { r = 255; g = 255; }
    Rgba([r, g, b, 255])
}

pub fn save_l1_plane_id_overlay(
    src_path: &Path,
    plane_id: &[u32],
    w: usize,
    h: usize,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 || plane_id.len() < w * h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = plane_id[idx];
            let (r, g, b) = if label == u32::MAX {
                (0u8, 0u8, 0u8)
            } else {
                let h = label.wrapping_mul(2654435761u32);
                (((h >> 16) & 255u32) as u8, ((h >> 8) & 255u32) as u8, (h & 255u32) as u8)
            };
            img.put_pixel(x as u32, y as u32, Rgba([r, g, b, 255]));
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l1_plane_id_overlay_{}.png", stem, time)))?;
    Ok(())
}

pub fn save_l1_plane_boundaries_on_src(
    src_path: &Path,
    plane_id: &[u32],
    w: usize,
    h: usize,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 || plane_id.len() < w * h {
        return Ok(());
    }
    let boundary = l1_boundary_mask(plane_id, w, h);
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    let color = Rgba([255, 255, 255, 255]);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if boundary[idx] != 0 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l1_plane_boundaries_on_src_{}.png", stem, time)))?;
    Ok(())
}

pub fn save_l1_edge4_overlay(
    src_path: &Path,
    edge4: &[u32],
    w: usize,
    h: usize,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 || edge4.len() < w * h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mask = edge4[idx] & 0xFu32;
            img.put_pixel(x as u32, y as u32, l1_mask_to_rgb(mask));
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l1_edge4_overlay_{}.png", stem, time)))?;
    Ok(())
}

pub fn save_l1_stop4_overlay(
    src_path: &Path,
    stop4: &[u32],
    w: usize,
    h: usize,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 || stop4.len() < w * h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mask = stop4[idx] & 0xFu32;
            img.put_pixel(x as u32, y as u32, l1_mask_to_rgb(mask));
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l1_stop4_overlay_{}.png", stem, time)))?;
    Ok(())
}

pub fn save_l1_plane_plus_edges(
    src_path: &Path,
    plane_id: &[u32],
    edge4: &[u32],
    stop4: &[u32],
    w: usize,
    h: usize,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 || plane_id.len() < w * h || edge4.len() < w * h || stop4.len() < w * h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = plane_id[idx];
            let (r, g, b) = if label == u32::MAX {
                (0u8, 0u8, 0u8)
            } else {
                let h = label.wrapping_mul(2654435761u32);
                (((h >> 16) & 255u32) as u8, ((h >> 8) & 255u32) as u8, (h & 255u32) as u8)
            };
            let mut color = Rgba([r, g, b, 255]);
            let mask = (edge4[idx] | stop4[idx]) & 0xFu32;
            if mask != 0 {
                color = Rgba([255, 255, 255, 255]);
            }
            img.put_pixel(x as u32, y as u32, color);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l1_plane_plus_edges_{}.png", stem, time)))?;
    Ok(())
}

pub fn save_layer2_overlay(
    src_path: &Path,
    pooled_mask: &[u32],
    out_w: usize,
    out_h: usize,
    _width: usize,
    _height: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if out_w == 0 || out_h == 0 {
        return Ok(());
    }
    if pooled_mask.len() < out_w * out_h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(out_w as u32, out_h as u32);
    let color_inactive = Rgba([128, 128, 128, 255]);
    let color_active = Rgba([255, 80, 80, 255]);
    for py in 0..out_h {
        for px in 0..out_w {
            let idx = py * out_w + px;
            let m = pooled_mask[idx];
            let active = (m & (1u32 << 1u32)) != 0u32;
            let color = if active { color_active } else { color_inactive };
            img.put_pixel(px as u32, py as u32, color);
        }
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer2_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn save_layer3_s_active_overlay(
    src_path: &Path,
    s_active: &[u32],
    out_w: usize,
    out_h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if out_w == 0 || out_h == 0 {
        return Ok(());
    }
    if s_active.len() < out_w * out_h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(out_w as u32, out_h as u32);
    let color_inactive = Rgba([0, 0, 0, 255]);
    let color_active = Rgba([255, 255, 255, 255]);
    for py in 0..out_h {
        for px in 0..out_w {
            let idx = py * out_w + px;
            let active = (s_active[idx] & 1u32) != 0u32;
            let color = if active { color_active } else { color_inactive };
            img.put_pixel(px as u32, py as u32, color);
        }
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer3_s_active_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn save_layer3_conn8_overlay(
    src_path: &Path,
    conn8: &[u32],
    out_w: usize,
    out_h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if out_w == 0 || out_h == 0 {
        return Ok(());
    }
    if conn8.len() < out_w * out_h {
        return Ok(());
    }
    let mut img = image::open(src_path)?.to_rgba8();
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    if orig_w == 0 || orig_h == 0 {
        return Ok(());
    }
    let colors = high_vis_dir_colors();
    let alpha = 160u8;
    for py in 0..out_h {
        for px in 0..out_w {
            let idx = py * out_w + px;
            let mask = conn8[idx] & 0xFFu32;
            if mask == 0 {
                continue;
            }
            let mut sum = [0u16; 3];
            let mut count = 0u16;
            for bit in 0..8u32 {
                if (mask & (1u32 << bit)) == 0 {
                    continue;
                }
                let c = colors[bit as usize];
                sum[0] += c.0[0] as u16;
                sum[1] += c.0[1] as u16;
                sum[2] += c.0[2] as u16;
                count += 1;
            }
            if count == 0 {
                continue;
            }
            let mixed = Rgba([
                (sum[0] / count) as u8,
                (sum[1] / count) as u8,
                (sum[2] / count) as u8,
                255,
            ]);
            let (sx0, sx1) = span_1d(px, out_w, orig_w);
            let (sy0, sy1) = span_1d(py, out_h, orig_h);
            blend_rect(&mut img, sx0, sy0, sx1, sy1, mixed, alpha);
        }
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer3_conn8_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn log_layer3_conn8(
    src_path: &Path,
    conn8: &[u32],
    out_w: usize,
    out_h: usize,
) -> std::io::Result<()> {
    if out_w == 0 || out_h == 0 {
        return Ok(());
    }
    if conn8.len() < out_w * out_h {
        return Ok(());
    }
    fs::create_dir_all("logs")?;
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let file = fs::File::create(format!("logs/l3_conn8_{}.log", stem))?;
    let mut w = BufWriter::new(file);
    for y in 0..out_h {
        for x in 0..out_w {
            let idx = y * out_w + x;
            let v = conn8[idx] & 0xFFu32;
            write!(w, "{:02X} ", v)?;
        }
        writeln!(w)?;
    }
    w.flush()
}

pub fn log_timing_layers(
    src_path: &Path,
    img_info: [u32; 4],
    l0: std::time::Duration,
    l1: std::time::Duration,
    l2: std::time::Duration,
    l3: std::time::Duration,
    l4: std::time::Duration,
    l5: std::time::Duration,
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
        writeln!(w, "layer3: {:?}", l3).ok();
        writeln!(w, "layer4: {:?}", l4).ok();
        writeln!(w, "layer5: {:?}", l5).ok();
        writeln!(w, "total: {:?}\n", total).ok();
        let _ = w.flush();
    }
}

pub fn save_layer4_expand_mask(
    src_path: &Path,
    meta: &[u32],
    w: usize,
    h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    save_layer4_channel(src_path, meta, w, h, tag, "expandMask", |v| ((v & 0xF_u32) * 17) as u8)
}

pub fn log_layer4_packed(
    src_path: &Path,
    bbox0: &[u32],
    bbox1: &[u32],
    meta: &[u32],
    w: usize,
    h: usize,
) -> std::io::Result<()> {
    if w == 0 || h == 0 {
        return Ok(());
    }
    let count = w * h;
    if bbox0.len() < count || bbox1.len() < count || meta.len() < count {
        return Ok(());
    }
    fs::create_dir_all("logs")?;
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let file = fs::File::create(format!("logs/l4_bboxes_{}.log", stem))?;
    let mut wtr = BufWriter::new(file);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let p0 = bbox0[idx];
            let p1 = bbox1[idx];
            let m = meta[idx] & 0xF_u32;
            if p0 == 0 && p1 == 0 && m == 0 {
                continue;
            }
            writeln!(wtr, "{} {} {:08X} {:08X} {:X}", x, y, p0, p1, m)?;
        }
    }
    wtr.flush()
}

pub fn save_layer4_bbox_overlay(
    src_path: &Path,
    bbox0: &[u32],
    bbox1: &[u32],
    meta: &[u32],
    grid_w: usize,
    grid_h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 {
        return Ok(());
    }
    let count = grid_w * grid_h;
    if meta.len() < count {
        return Ok(());
    }
    let mut img = image::open(src_path)?.to_rgba8();
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    if orig_w == 0 || orig_h == 0 {
        return Ok(());
    }
    let dir_colors = [
        Rgba([0, 255, 255, 255]), // N  cyan
        Rgba([255, 0, 255, 255]), // E  magenta
        Rgba([255, 160, 0, 255]), // S  orange
        Rgba([0, 128, 255, 255]), // W  blue
    ];
    for idx in 0..count {
        let mask = meta[idx] & 0xF;
        let x = idx % grid_w;
        let y = idx / grid_w;
        let (x0p, x1p) = span_1d(x, grid_w, orig_w);
        let (y0p, y1p) = span_1d(y, grid_h, orig_h);
        if mask == 0 {
            draw_rect_outline(&mut img, x0p, y0p, x1p, y1p, Rgba([160, 160, 160, 255]));
            continue;
        }
        if (mask & 0x1) != 0 {
            for x in x0p..=x1p {
                img.put_pixel(x as u32, y0p as u32, dir_colors[0]);
            }
        }
        if (mask & 0x2) != 0 {
            for y in y0p..=y1p {
                img.put_pixel(x1p as u32, y as u32, dir_colors[1]);
            }
        }
        if (mask & 0x4) != 0 {
            for x in x0p..=x1p {
                img.put_pixel(x as u32, y1p as u32, dir_colors[2]);
            }
        }
        if (mask & 0x8) != 0 {
            for y in y0p..=y1p {
                img.put_pixel(x0p as u32, y as u32, dir_colors[3]);
            }
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer4_bbox_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

fn save_layer4_channel<F: Fn(u32) -> u8>(
    src_path: &Path,
    data: &[u32],
    w: usize,
    h: usize,
    tag: &str,
    name: &str,
    map: F,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 {
        return Ok(());
    }
    if data.len() < w * h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let v = map(data[idx] & 0xFFFF);
            img.put_pixel(x as u32, y as u32, Rgba([v, v, v, 255]));
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer4_{}_{}_{}.png", stem, name, tag, time)))?;
    Ok(())
}

pub fn save_layer5_labels(
    src_path: &Path,
    labels: &[u32],
    w: usize,
    h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 {
        return Ok(());
    }
    if labels.len() < w * h {
        return Ok(());
    }
    let mut img = image::open(src_path)?.to_rgba8();
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    if orig_w == 0 || orig_h == 0 {
        return Ok(());
    }
    let alpha = 140u8;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = labels[idx];
            let v = if label == u32::MAX {
                0u8
            } else {
                ((label.wrapping_mul(2654435761u32)) >> 24) as u8
            };
            let (sx0, sx1) = span_1d(x, w, orig_w);
            let (sy0, sy1) = span_1d(y, h, orig_h);
            blend_rect(&mut img, sx0, sy0, sx1, sy1, Rgba([v, v, v, 255]), alpha);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer5_labels_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn save_layer5_accum_mask(
    src_path: &Path,
    labels: &[u32],
    acc_count: &[u32],
    w: usize,
    h: usize,
    min_count: u32,
    tag: &str,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 {
        return Ok(());
    }
    if labels.len() < w * h {
        return Ok(());
    }
    let mut img = image::open(src_path)?.to_rgba8();
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    if orig_w == 0 || orig_h == 0 {
        return Ok(());
    }
    let hit = Rgba([255, 64, 64, 255]);
    let alpha = 160u8;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = labels[idx];
            let mut on = false;
            if label != u32::MAX {
                let li = label as usize;
                if li < acc_count.len() && acc_count[li] >= min_count {
                    on = true;
                }
            }
            let (sx0, sx1) = span_1d(x, w, orig_w);
            let (sy0, sy1) = span_1d(y, h, orig_h);
            if on {
                blend_rect(&mut img, sx0, sy0, sx1, sy1, hit, alpha);
            }
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_layer5_accum_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn save_l5_score_map(
    src_path: &Path,
    score: &[u32],
    tile_w: usize,
    tile_h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if tile_w == 0 || tile_h == 0 {
        return Ok(());
    }
    if score.len() < tile_w * tile_h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(tile_w as u32, tile_h as u32);
    for y in 0..tile_h {
        for x in 0..tile_w {
            let idx = y * tile_w + x;
            let v = score[idx].min(255) as u8;
            img.put_pixel(x as u32, y as u32, Rgba([v, v, v, 255]));
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l5_score_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn save_l5_tile_keep(
    src_path: &Path,
    keep: &[u32],
    tile_w: usize,
    tile_h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if tile_w == 0 || tile_h == 0 {
        return Ok(());
    }
    if keep.len() < tile_w * tile_h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(tile_w as u32, tile_h as u32);
    for y in 0..tile_h {
        for x in 0..tile_w {
            let idx = y * tile_w + x;
            let v = if keep[idx] != 0 { 255u8 } else { 0u8 };
            img.put_pixel(x as u32, y as u32, Rgba([v, v, v, 255]));
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l5_tilekeep_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn save_l5_roi_mask(
    src_path: &Path,
    roi: &[u32],
    w: usize,
    h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 {
        return Ok(());
    }
    if roi.len() < w * h {
        return Ok(());
    }
    let mut img = image::open(src_path)?.to_rgba8();
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    if orig_w == 0 || orig_h == 0 {
        return Ok(());
    }
    let color = Rgba([255, 255, 255, 255]);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if roi[idx] == 0 {
                continue;
            }
            let left = if x > 0 { roi[idx - 1] } else { 0 };
            let right = if x + 1 < w { roi[idx + 1] } else { 0 };
            let up = if y > 0 { roi[idx - w] } else { 0 };
            let down = if y + 1 < h { roi[idx + w] } else { 0 };
            let is_edge = left == 0 || right == 0 || up == 0 || down == 0;
            if !is_edge {
                continue;
            }
            let (sx0, sx1) = span_1d(x, w, orig_w);
            let (sy0, sy1) = span_1d(y, h, orig_h);
            draw_rect_outline(&mut img, sx0, sy0, sx1, sy1, color);
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l5_roi_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn save_l5_label_map(
    src_path: &Path,
    labels: &[u32],
    w: usize,
    h: usize,
    tag: &str,
) -> Result<(), image::ImageError> {
    if w == 0 || h == 0 {
        return Ok(());
    }
    if labels.len() < w * h {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = labels[idx];
            let v = if label == u32::MAX {
                0u8
            } else {
                ((label.wrapping_mul(2654435761u32)) >> 24) as u8
            };
            img.put_pixel(x as u32, y as u32, Rgba([v, v, v, 255]));
        }
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l5_labels_{}_{}.png", stem, tag, time)))?;
    Ok(())
}
