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
    pub save_l1_bbox: bool,
    pub save_l1_bbox_stride2: bool,
    pub log_timing: bool,
    pub l0_edge_th_r: u32,
    pub l0_edge_th_g: u32,
    pub l0_edge_th_b: u32,
    pub l0_dir_min_channels: u32,
    pub l0_pixel_min_dirs: u32,
    pub l1_enable_stride2: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            max_images: usize::MAX,
            test_image: None,
            images_dir: None,
            save_layer0: false,
            save_l1_bbox: false,
            save_l1_bbox_stride2: false,
            log_timing: false,
            l0_edge_th_r: 15,
            l0_edge_th_g: 15,
            l0_edge_th_b: 20,
            l0_dir_min_channels: 3,
            l0_pixel_min_dirs: 1,
            l1_enable_stride2: false,
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
            "SAVE_L1_BBOX" => cfg.save_l1_bbox = parse_bool(value),
            "SAVE_L1_BBOX_STRIDE2" => cfg.save_l1_bbox_stride2 = parse_bool(value),
            "LOG_TIMING" => cfg.log_timing = parse_bool(value),
            "L0_EDGE_TH_R" => cfg.l0_edge_th_r = value.parse().unwrap_or(cfg.l0_edge_th_r),
            "L0_EDGE_TH_G" => cfg.l0_edge_th_g = value.parse().unwrap_or(cfg.l0_edge_th_g),
            "L0_EDGE_TH_B" => cfg.l0_edge_th_b = value.parse().unwrap_or(cfg.l0_edge_th_b),
            "L0_DIR_MIN_CHANNELS" => cfg.l0_dir_min_channels = value.parse().unwrap_or(cfg.l0_dir_min_channels),
            "L0_PIXEL_MIN_DIRS" => cfg.l0_pixel_min_dirs = value.parse().unwrap_or(cfg.l0_pixel_min_dirs),
            "L1_ENABLE_STRIDE2" => cfg.l1_enable_stride2 = parse_bool(value),
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
                if in_layer1 {
                    match key_lower.as_str() {
                        "enable_stride2" => cfg.l1_enable_stride2 = parse_bool(value),
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
    apply_env_bool("SAVE_L1_BBOX", &mut cfg.save_l1_bbox);
    apply_env_bool("SAVE_L1_BBOX_STRIDE2", &mut cfg.save_l1_bbox_stride2);
    apply_env_bool("LOG_TIMING", &mut cfg.log_timing);
    apply_env_u32("L0_EDGE_TH_R", &mut cfg.l0_edge_th_r);
    apply_env_u32("L0_EDGE_TH_G", &mut cfg.l0_edge_th_g);
    apply_env_u32("L0_EDGE_TH_B", &mut cfg.l0_edge_th_b);
    apply_env_u32("L0_DIR_MIN_CHANNELS", &mut cfg.l0_dir_min_channels);
    apply_env_u32("L0_PIXEL_MIN_DIRS", &mut cfg.l0_pixel_min_dirs);
    apply_env_bool("L1_ENABLE_STRIDE2", &mut cfg.l1_enable_stride2);
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
