use image::{Rgba, RgbaImage};
use chrono::Local;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

// model.rs에서 정의된 타입을 사용합니다.
pub const SAVE_DIR: &str = "result_images";

// ======= 보조 함수 (밀림 방지 매핑) =======

fn local_timestamp_uk() -> String {
    Local::now().format("%d-%m-%Y_%H-%M-%S").to_string()
}

#[inline]
pub fn span_1d(cell: usize, grid: usize, orig: usize) -> (usize, usize) {
    if grid == 0 || orig == 0 {
        return (0, 0);
    }
    let s0 = (cell * orig) / grid;
    let s1 = (((cell + 1) * orig) / grid).saturating_sub(1);
    (s0.min(orig - 1), s1.min(orig - 1))
}

#[inline]
fn span_box(coord0: u32, coord1: u32, grid: usize, orig: usize) -> (usize, usize) {
    if grid == 0 || orig == 0 {
        return (0, 0);
    }
    let s0 = (coord0 as usize * orig) / grid;
    let s1 = ((coord1 as usize * orig) / grid).saturating_sub(1);
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

    let w_i32 = (w - 1) as i32;
    let h_i32 = (h - 1) as i32;
    let corners = [
        (sx0 as i32, sy0 as i32),
        (sx1 as i32, sy0 as i32),
        (sx0 as i32, sy1 as i32),
        (sx1 as i32, sy1 as i32),
    ];
    for (cx, cy) in corners {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let px = (cx + dx).clamp(0, w_i32) as u32;
                let py = (cy + dy).clamp(0, h_i32) as u32;
                img.put_pixel(px, py, color);
            }
        }
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

    let w = orig_w as i32;
    let h = orig_h as i32;

    let color_active = Rgba([255, 80, 80, 255]);
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

pub fn save_layer3_pass1_anchor_overlay(
    src_path: &Path,
    out_w: usize,
    out_h: usize,
    aw: usize,
    ah: usize,
    anchor_bbox: &[u32],
    anchor_score: &[u32],
    anchor_meta: &[u32],
) -> Result<(), image::ImageError> {
    if out_w == 0 || out_h == 0 || aw == 0 || ah == 0 {
        return Ok(());
    }
    let anchor_count = aw * ah;
    if anchor_bbox.len() < anchor_count * 2 || anchor_score.len() < anchor_count || anchor_meta.len() < anchor_count {
        return Ok(());
    }

    let mut img = image::open(src_path)?.to_rgba8();
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    if orig_w == 0 || orig_h == 0 {
        return Ok(());
    }

    let color_valid = Rgba([0, 255, 0, 255]);
    let color_boundary = Rgba([255, 128, 0, 255]);

    for idx in 0..anchor_count {
        let meta = anchor_meta[idx];
        if (meta & 1u32) == 0u32 {
            continue;
        }
        let bbox0 = anchor_bbox[idx * 2];
        let bbox1 = anchor_bbox[idx * 2 + 1];
        let x0 = (bbox0 & 0xFFFF) as u32;
        let y0 = (bbox0 >> 16) as u32;
        let x1 = (bbox1 & 0xFFFF) as u32;
        let y1 = (bbox1 >> 16) as u32;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let (sx0, sx1) = span_box(x0, x1, out_w, orig_w);
        let (sy0, sy1) = span_box(y0, y1, out_h, orig_h);
        let ended_by_boundary = (meta & 2u32) != 0u32;
        let color = if ended_by_boundary { color_boundary } else { color_valid };
        let _score = anchor_score[idx];
        draw_rect_outline(&mut img, sx0, sy0, sx1, sy1, color);
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l3_pass1_anchors_{}.png", stem, time)))?;
    Ok(())
}

pub fn save_layer3_pass2_block_topk_overlay(
    src_path: &Path,
    out_w: usize,
    out_h: usize,
    bw: usize,
    bh: usize,
    block_boxes: &[u32],
    block_scores: &[u32],
    block_valid: &[u32],
) -> Result<(), image::ImageError> {
    if out_w == 0 || out_h == 0 || bw == 0 || bh == 0 {
        return Ok(());
    }
    let block_count = bw * bh;
    let slot_count = block_count * 16;
    if block_boxes.len() < slot_count * 2 || block_scores.len() < slot_count || block_valid.len() < slot_count {
        return Ok(());
    }

    let mut img = image::open(src_path)?.to_rgba8();
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    if orig_w == 0 || orig_h == 0 {
        return Ok(());
    }

    let colors = [
        Rgba([255, 0, 0, 255]),
        Rgba([255, 128, 0, 255]),
        Rgba([255, 255, 0, 255]),
        Rgba([0, 255, 0, 255]),
        Rgba([0, 128, 255, 255]),
    ];

    for idx in 0..slot_count {
        if (block_valid[idx] & 1u32) == 0u32 {
            continue;
        }
        let bidx = idx * 2;
        let bbox0 = block_boxes[bidx];
        let bbox1 = block_boxes[bidx + 1];
        let x0 = (bbox0 & 0xFFFF) as u32;
        let y0 = (bbox0 >> 16) as u32;
        let x1 = (bbox1 & 0xFFFF) as u32;
        let y1 = (bbox1 >> 16) as u32;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let (sx0, sx1) = span_box(x0, x1, out_w, orig_w);
        let (sy0, sy1) = span_box(y0, y1, out_h, orig_h);
        let rank = idx % 5;
        let _score = block_scores[idx];
        draw_rect_outline(&mut img, sx0, sy0, sx1, sy1, colors[rank]);
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l3_pass2_top5_{}.png", stem, time)))?;
    Ok(())
}

pub fn save_layer3_block_overlay(
    src_path: &Path,
    pooled_mask: &[u32],
    out_w: usize,
    out_h: usize,
    block_boxes: &[u32],
    block_valid: &[u32],
    tag: &str,
) -> Result<(), image::ImageError> {
    if out_w == 0 || out_h == 0 {
        return Ok(());
    }
    let slot_count = block_valid.len();
    if block_boxes.len() < slot_count * 2 {
        return Ok(());
    }

    let mut img = image::RgbaImage::new(out_w as u32, out_h as u32);
    let color_active = Rgba([200, 200, 200, 255]);
    let color_box = Rgba([0, 200, 255, 255]);
    let color_boundary = Rgba([255, 128, 0, 255]);
    const FLAG_TOUCH_N: u32 = 16u32;
    const FLAG_TOUCH_E: u32 = 32u32;
    const FLAG_TOUCH_S: u32 = 64u32;
    const FLAG_TOUCH_W: u32 = 128u32;

    if pooled_mask.len() >= out_w * out_h {
        for y in 0..out_h {
            for x in 0..out_w {
                let idx = y * out_w + x;
                if (pooled_mask[idx] & (1u32 << 1u32)) != 0u32 {
                    img.put_pixel(x as u32, y as u32, color_active);
                }
            }
        }
    }

    for idx in 0..slot_count {
        let flags = block_valid[idx];
        if (flags & 1u32) == 0u32 {
            continue;
        }
        let bidx = idx * 2;
        let b0 = block_boxes[bidx];
        let b1 = block_boxes[bidx + 1];
        let x0 = (b0 & 0xFFFF) as u32;
        let y0 = (b0 >> 16) as u32;
        let x1 = (b1 & 0xFFFF) as u32;
        let y1 = (b1 >> 16) as u32;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let sx0 = (x0 as usize).min(out_w - 1);
        let sy0 = (y0 as usize).min(out_h - 1);
        let sx1 = (x1.saturating_sub(1) as usize).min(out_w - 1);
        let sy1 = (y1.saturating_sub(1) as usize).min(out_h - 1);
        let touched = (flags & (FLAG_TOUCH_N | FLAG_TOUCH_E | FLAG_TOUCH_S | FLAG_TOUCH_W)) != 0u32;
        let color = if touched { color_boundary } else { color_box };
        draw_rect_outline(&mut img, sx0, sy0, sx1, sy1, color);
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l3_block_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

pub fn save_layer4_boxes_overlay(
    src_path: &Path,
    pooled_mask: &[u32],
    grid_w: usize,
    grid_h: usize,
    boxes: &[(u32, u32, u32, u32)],
    tag: &str,
) -> Result<(), image::ImageError> {
    if grid_w == 0 || grid_h == 0 {
        return Ok(());
    }
    let mut img = image::RgbaImage::new(grid_w as u32, grid_h as u32);
    let color_active = Rgba([200, 200, 200, 255]);
    let color_box = Rgba([0, 200, 255, 255]);

    if pooled_mask.len() >= grid_w * grid_h {
        for y in 0..grid_h {
            for x in 0..grid_w {
                let idx = y * grid_w + x;
                if (pooled_mask[idx] & (1u32 << 1u32)) != 0u32 {
                    img.put_pixel(x as u32, y as u32, color_active);
                }
            }
        }
    }

    for &(x0, y0, x1, y1) in boxes {
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let sx0 = (x0 as usize).min(grid_w - 1);
        let sy0 = (y0 as usize).min(grid_h - 1);
        let sx1 = (x1.saturating_sub(1) as usize).min(grid_w - 1);
        let sy1 = (y1.saturating_sub(1) as usize).min(grid_h - 1);
        draw_rect_outline(&mut img, sx0, sy0, sx1, sy1, color_box);
    }

    fs::create_dir_all(SAVE_DIR).map_err(image::ImageError::IoError)?;
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    img.save(Path::new(SAVE_DIR).join(format!("{}_l4_{}_{}.png", stem, tag, time)))?;
    Ok(())
}

// ======= CSV 및 데이터 로그 관련 함수 =======

pub fn log_timing_block(
    src_path: &Path,
    img_info: [u32; 4],
    l0: std::time::Duration,
    l1: std::time::Duration,
    l2: std::time::Duration,
    l3: std::time::Duration,
    l4: std::time::Duration,
    total: std::time::Duration,
) {
    static LOG_WRITER: OnceLock<Mutex<BufWriter<std::fs::File>>> = OnceLock::new();
    let writer = LOG_WRITER.get_or_init(|| {
        let ts = local_timestamp_uk();
        let file = OpenOptions::new().create(true).append(true).open(format!("object_detection_{}.log", ts)).unwrap();
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
        writeln!(w, "total: {:?}", total).ok();
        w.flush().ok();
    }
}
