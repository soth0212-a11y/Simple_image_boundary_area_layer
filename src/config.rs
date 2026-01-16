use std::fs;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub max_images: usize,
    pub test_image: Option<PathBuf>,
    pub images_dir: Option<PathBuf>,
    pub save_layer0: bool,
    pub save_layer1: bool,
    pub save_layer2: bool,
    pub save_l2_boundary: bool,
    pub save_l2_labels: bool,
    pub save_l2_bboxes: bool,
    pub save_layer3: bool,
    pub log_layer3: bool,
    pub log_timing: bool,
    pub save_layer4: bool,
    pub save_layer5_labels: bool,
    pub save_layer5_accum: bool,
    pub save_layer5_merged: bool,
    pub save_l5_debug: bool,
    pub l1_iter: u32,
    pub l2_iter: u32,
    pub l2_edge_cell_th: u32,
    pub l2_area_th: u32,
    pub l2_max_boxes: u32,
    pub l0_edge_th_r: u32,
    pub l0_edge_th_g: u32,
    pub l0_edge_th_b: u32,
    pub l0_dir_min_channels: u32,
    pub l0_pixel_min_dirs: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            max_images: usize::MAX,
            test_image: None,
            images_dir: None,
            save_layer0: false,
            save_layer1: false,
            save_layer2: false,
            save_l2_boundary: false,
            save_l2_labels: false,
            save_l2_bboxes: false,
            save_layer3: false,
            log_layer3: false,
            log_timing: false,
            save_layer4: false,
            save_layer5_labels: false,
            save_layer5_accum: false,
            save_layer5_merged: false,
            save_l5_debug: false,
            l1_iter: 0,
            l2_iter: 0,
            l2_edge_cell_th: 4,
            l2_area_th: 4,
            l2_max_boxes: 8192,
            l0_edge_th_r: 15,
            l0_edge_th_g: 15,
            l0_edge_th_b: 20,
            l0_dir_min_channels: 3,
            l0_pixel_min_dirs: 1,
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
            "SAVE_LAYER1" => cfg.save_layer1 = parse_bool(value),
            "SAVE_LAYER2" => cfg.save_layer2 = parse_bool(value),
            "SAVE_L2_BOUNDARY" => cfg.save_l2_boundary = parse_bool(value),
            "SAVE_L2_LABELS" => cfg.save_l2_labels = parse_bool(value),
            "SAVE_L2_BBOXES" => cfg.save_l2_bboxes = parse_bool(value),
            "SAVE_LAYER3" => cfg.save_layer3 = parse_bool(value),
            "LOG_LAYER3" => cfg.log_layer3 = parse_bool(value),
            "LOG_TIMING" => cfg.log_timing = parse_bool(value),
            "SAVE_LAYER4" => cfg.save_layer4 = parse_bool(value),
            "SAVE_L5_LABELS" => cfg.save_layer5_labels = parse_bool(value),
            "SAVE_L5_ACCUM" => cfg.save_layer5_accum = parse_bool(value),
            "SAVE_L5_MERGED" => cfg.save_layer5_merged = parse_bool(value),
            "SAVE_L5_DEBUG" => cfg.save_l5_debug = parse_bool(value),
            "L1_ITER" => cfg.l1_iter = value.parse().unwrap_or(cfg.l1_iter),
            "L2_ITER" => cfg.l2_iter = value.parse().unwrap_or(cfg.l2_iter),
            "L2_EDGE_CELL_TH" => cfg.l2_edge_cell_th = value.parse().unwrap_or(cfg.l2_edge_cell_th),
            "L2_AREA_TH" => cfg.l2_area_th = value.parse().unwrap_or(cfg.l2_area_th),
            "L2_MAX_BOXES" => cfg.l2_max_boxes = value.parse().unwrap_or(cfg.l2_max_boxes),
            "L0_EDGE_TH_R" => cfg.l0_edge_th_r = value.parse().unwrap_or(cfg.l0_edge_th_r),
            "L0_EDGE_TH_G" => cfg.l0_edge_th_g = value.parse().unwrap_or(cfg.l0_edge_th_g),
            "L0_EDGE_TH_B" => cfg.l0_edge_th_b = value.parse().unwrap_or(cfg.l0_edge_th_b),
            "L0_DIR_MIN_CHANNELS" => cfg.l0_dir_min_channels = value.parse().unwrap_or(cfg.l0_dir_min_channels),
            "L0_PIXEL_MIN_DIRS" => cfg.l0_pixel_min_dirs = value.parse().unwrap_or(cfg.l0_pixel_min_dirs),
            _ => {
                if in_layer1 {
                    match key_lower.as_str() {
                        "iter" => cfg.l1_iter = value.parse().unwrap_or(cfg.l1_iter),
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
                        "iter" => cfg.l2_iter = value.parse().unwrap_or(cfg.l2_iter),
                        "edge_cell_th" => cfg.l2_edge_cell_th = value.parse().unwrap_or(cfg.l2_edge_cell_th),
                        "area_th" => cfg.l2_area_th = value.parse().unwrap_or(cfg.l2_area_th),
                        "max_boxes" => cfg.l2_max_boxes = value.parse().unwrap_or(cfg.l2_max_boxes),
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
    apply_env_bool("SAVE_LAYER1", &mut cfg.save_layer1);
    apply_env_bool("SAVE_LAYER2", &mut cfg.save_layer2);
    apply_env_bool("SAVE_L2_BOUNDARY", &mut cfg.save_l2_boundary);
    apply_env_bool("SAVE_L2_LABELS", &mut cfg.save_l2_labels);
    apply_env_bool("SAVE_L2_BBOXES", &mut cfg.save_l2_bboxes);
    apply_env_bool("SAVE_LAYER3", &mut cfg.save_layer3);
    apply_env_bool("LOG_LAYER3", &mut cfg.log_layer3);
    apply_env_bool("LOG_TIMING", &mut cfg.log_timing);
    apply_env_bool("SAVE_LAYER4", &mut cfg.save_layer4);
    apply_env_bool("SAVE_L5_LABELS", &mut cfg.save_layer5_labels);
    apply_env_bool("SAVE_L5_ACCUM", &mut cfg.save_layer5_accum);
    apply_env_bool("SAVE_L5_MERGED", &mut cfg.save_layer5_merged);
    apply_env_bool("SAVE_L5_DEBUG", &mut cfg.save_l5_debug);
    apply_env_u32("L1_ITER", &mut cfg.l1_iter);
    apply_env_u32("L2_ITER", &mut cfg.l2_iter);
    apply_env_u32("L2_EDGE_CELL_TH", &mut cfg.l2_edge_cell_th);
    apply_env_u32("L2_AREA_TH", &mut cfg.l2_area_th);
    apply_env_u32("L2_MAX_BOXES", &mut cfg.l2_max_boxes);
    apply_env_u32("L1_EDGE_TH_R", &mut cfg.l0_edge_th_r);
    apply_env_u32("L1_EDGE_TH_G", &mut cfg.l0_edge_th_g);
    apply_env_u32("L1_EDGE_TH_B", &mut cfg.l0_edge_th_b);
    apply_env_u32("L1_DIR_MIN_CHANNELS", &mut cfg.l0_dir_min_channels);
    apply_env_u32("L1_PIXEL_MIN_DIRS", &mut cfg.l0_pixel_min_dirs);

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
