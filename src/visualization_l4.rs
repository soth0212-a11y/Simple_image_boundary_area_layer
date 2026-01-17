use crate::rle_ccl_gpu::OutBox;
use image::{Rgba, RgbaImage};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const SAVE_DIR: &str = "result_images";

#[inline]
fn span_1d(cell: usize, grid: usize, orig: usize) -> (usize, usize) {
    if grid == 0 || orig == 0 {
        return (0, 0);
    }
    let s0 = (cell * orig) / grid;
    let s1 = (((cell + 1) * orig) / grid).saturating_sub(1);
    (s0.min(orig - 1), s1.min(orig - 1))
}

#[inline]
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

#[inline]
fn draw_rect_outline(img: &mut RgbaImage, x0: usize, y0: usize, x1: usize, y1: usize, color: Rgba<u8>) {
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

pub fn save_l4_overlay(
    src_path: &Path,
    boxes: &[OutBox],
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
        if max_x <= min_x || max_y <= min_y {
            continue;
        }
        let color = color565_to_rgb(b.color565);
        draw_rect_outline(&mut img, min_x, min_y, max_x, max_y, color);
    }
    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l4_{}_{}.png", stem, name_suffix, time)))?;
    Ok(())
}

