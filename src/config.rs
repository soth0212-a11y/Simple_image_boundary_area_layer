use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub max_images: usize,
    pub test_image: Option<PathBuf>,
    pub images_dir: Option<PathBuf>,
    pub save_layer0: bool,
    pub save_l1_segments: bool,
    pub save_l2_boxes: bool,
    pub log_timing: bool,
    pub l0_edge_th_r: u32,
    pub l0_edge_th_g: u32,
    pub l0_edge_th_b: u32,
    pub l0_dir_min_channels: u32,
    pub l0_pixel_min_dirs: u32,
    pub l2_max_out: u32,
    pub l2_min_w: u32,
    pub l2_min_h: u32,
    pub l2_color_tol: u32,
    pub l2_gap_x: u32,
    pub l2_gap_y: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            max_images: usize::MAX,
            test_image: None,
            images_dir: None,
            save_layer0: false,
            save_l1_segments: false,
            save_l2_boxes: false,
            log_timing: false,
            l0_edge_th_r: 15,
            l0_edge_th_g: 15,
            l0_edge_th_b: 20,
            l0_dir_min_channels: 3,
            l0_pixel_min_dirs: 1,
            l2_max_out: 1_000_000,
            l2_min_w: 2,
            l2_min_h: 2,
            l2_color_tol: 20,
            l2_gap_x: 1,
            l2_gap_y: 1,
        }
    }
}

static CONFIG: OnceLock<AppConfig> = OnceLock::new();

pub fn init() -> AppConfig {
    let mut cfg = load_from_paths(&[Path::new("setthing.conf"), Path::new("setting.conf")]);
    apply_env_overrides(&mut cfg);
    let _ = CONFIG.set(cfg.clone());
    cfg
}

pub fn get() -> &'static AppConfig {
    CONFIG.get().expect("config not initialized")
}

fn load_from_paths(paths: &[&Path]) -> AppConfig {
    for path in paths {
        if let Ok(text) = fs::read_to_string(path) {
            return parse_config(&text);
        }
    }
    AppConfig::default()
}

fn parse_config(text: &str) -> AppConfig {
    let mut cfg = AppConfig::default();
    let mut section: Option<&str> = None;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = Some(line.trim_matches(&['[', ']'][..]).trim());
            continue;
        }
        let (key, value) = match line.split_once('=') {
            Some(v) => v,
            None => continue,
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"');
        let section = section.unwrap_or("");
        let key_lower = key.to_ascii_lowercase();
        let in_layer0 = section.eq_ignore_ascii_case("Layer0");
        let in_layer1 = section.eq_ignore_ascii_case("Layer1");
        let in_layer2 = section.eq_ignore_ascii_case("Layer2");
        match key {
            "MAX_IMAGES" => {
                if let Ok(v) = value.parse::<usize>() {
                    cfg.max_images = if v == 0 { usize::MAX } else { v };
                }
            }
            "TEST_IMAGE" => {
                if !value.is_empty() {
                    cfg.test_image = Some(PathBuf::from(value));
                }
            }
            "IMAGES_DIR" => {
                if !value.is_empty() {
                    cfg.images_dir = Some(PathBuf::from(value));
                }
            }
            "SAVE_LAYER0" => cfg.save_layer0 = parse_bool(value),
            "SAVE_L1_SEGMENTS" => cfg.save_l1_segments = parse_bool(value),
            "SAVE_L2_BOXES" => cfg.save_l2_boxes = parse_bool(value),
            "LOG_TIMING" => cfg.log_timing = parse_bool(value),
            "L0_EDGE_TH_R" => cfg.l0_edge_th_r = value.parse().unwrap_or(cfg.l0_edge_th_r),
            "L0_EDGE_TH_G" => cfg.l0_edge_th_g = value.parse().unwrap_or(cfg.l0_edge_th_g),
            "L0_EDGE_TH_B" => cfg.l0_edge_th_b = value.parse().unwrap_or(cfg.l0_edge_th_b),
            "L0_DIR_MIN_CHANNELS" => cfg.l0_dir_min_channels = value.parse().unwrap_or(cfg.l0_dir_min_channels),
            "L0_PIXEL_MIN_DIRS" => cfg.l0_pixel_min_dirs = value.parse().unwrap_or(cfg.l0_pixel_min_dirs),
            "L2_MAX_OUT" => cfg.l2_max_out = value.parse().unwrap_or(cfg.l2_max_out),
            "L2_MIN_W" => cfg.l2_min_w = value.parse().unwrap_or(cfg.l2_min_w),
            "L2_MIN_H" => cfg.l2_min_h = value.parse().unwrap_or(cfg.l2_min_h),
            "L2_COLOR_TOL" => cfg.l2_color_tol = value.parse().unwrap_or(cfg.l2_color_tol),
            "L2_GAP_X" => cfg.l2_gap_x = value.parse().unwrap_or(cfg.l2_gap_x),
            "L2_GAP_Y" => cfg.l2_gap_y = value.parse().unwrap_or(cfg.l2_gap_y),
            _ => {
                if in_layer0 {
                    match key_lower.as_str() {
                        "edge_th_r" => cfg.l0_edge_th_r = value.parse().unwrap_or(cfg.l0_edge_th_r),
                        "edge_th_g" => cfg.l0_edge_th_g = value.parse().unwrap_or(cfg.l0_edge_th_g),
                        "edge_th_b" => cfg.l0_edge_th_b = value.parse().unwrap_or(cfg.l0_edge_th_b),
                        "dir_min_channels" => cfg.l0_dir_min_channels = value.parse().unwrap_or(cfg.l0_dir_min_channels),
                        "pixel_min_dirs" => cfg.l0_pixel_min_dirs = value.parse().unwrap_or(cfg.l0_pixel_min_dirs),
                        _ => {}
                    }
                }
                if in_layer2 {
                    match key_lower.as_str() {
                        "max_out" => cfg.l2_max_out = value.parse().unwrap_or(cfg.l2_max_out),
                        "min_w" => cfg.l2_min_w = value.parse().unwrap_or(cfg.l2_min_w),
                        "min_h" => cfg.l2_min_h = value.parse().unwrap_or(cfg.l2_min_h),
                        "color_tol" => cfg.l2_color_tol = value.parse().unwrap_or(cfg.l2_color_tol),
                        "gap_x" => cfg.l2_gap_x = value.parse().unwrap_or(cfg.l2_gap_x),
                        "gap_y" => cfg.l2_gap_y = value.parse().unwrap_or(cfg.l2_gap_y),
                        _ => {}
                    }
                }
            }
        }
    }
    cfg
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn apply_env_overrides(cfg: &mut AppConfig) {
    if let Ok(v) = env::var("MAX_IMAGES") {
        if let Ok(parsed) = v.parse::<usize>() {
            cfg.max_images = if parsed == 0 { usize::MAX } else { parsed };
        }
    }
    if let Ok(v) = env::var("TEST_IMAGE") {
        if !v.trim().is_empty() {
            cfg.test_image = Some(PathBuf::from(v));
        }
    }
    if let Ok(v) = env::var("IMAGES_DIR") {
        if !v.trim().is_empty() {
            cfg.images_dir = Some(PathBuf::from(v));
        }
    }

    apply_env_bool("SAVE_LAYER0", &mut cfg.save_layer0);
    apply_env_bool("SAVE_L1_SEGMENTS", &mut cfg.save_l1_segments);
    apply_env_bool("SAVE_L2_BOXES", &mut cfg.save_l2_boxes);
    apply_env_bool("LOG_TIMING", &mut cfg.log_timing);
    apply_env_u32("L0_EDGE_TH_R", &mut cfg.l0_edge_th_r);
    apply_env_u32("L0_EDGE_TH_G", &mut cfg.l0_edge_th_g);
    apply_env_u32("L0_EDGE_TH_B", &mut cfg.l0_edge_th_b);
    apply_env_u32("L0_DIR_MIN_CHANNELS", &mut cfg.l0_dir_min_channels);
    apply_env_u32("L0_PIXEL_MIN_DIRS", &mut cfg.l0_pixel_min_dirs);
    apply_env_u32("L2_MAX_OUT", &mut cfg.l2_max_out);
    apply_env_u32("L2_MIN_W", &mut cfg.l2_min_w);
    apply_env_u32("L2_MIN_H", &mut cfg.l2_min_h);
    apply_env_u32("L2_COLOR_TOL", &mut cfg.l2_color_tol);
    apply_env_u32("L2_GAP_X", &mut cfg.l2_gap_x);
    apply_env_u32("L2_GAP_Y", &mut cfg.l2_gap_y);
}

fn apply_env_bool(name: &str, target: &mut bool) {
    if let Ok(v) = env::var(name) {
        *target = parse_bool(&v);
    }
}

fn apply_env_u32(name: &str, target: &mut u32) {
    if let Ok(v) = env::var(name) {
        if let Ok(parsed) = v.parse::<u32>() {
            *target = parsed;
        }
    }
}
