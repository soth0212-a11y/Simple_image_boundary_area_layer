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
    pub save_layer3: bool,
    pub save_layer4: bool,
    pub log_timing: bool,
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
            save_layer3: false,
            save_layer4: false,
            log_timing: true,
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
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = match line.split_once('=') {
            Some(v) => v,
            None => continue,
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"');
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
            "SAVE_LAYER3" => cfg.save_layer3 = parse_bool(value),
            "SAVE_LAYER4" => cfg.save_layer4 = parse_bool(value),
            "LOG_TIMING" => cfg.log_timing = parse_bool(value),
            _ => {}
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
    apply_env_bool("SAVE_LAYER3", &mut cfg.save_layer3);
    apply_env_bool("SAVE_LAYER4", &mut cfg.save_layer4);
    apply_env_bool("LOG_TIMING", &mut cfg.log_timing);
}

fn apply_env_bool(name: &str, target: &mut bool) {
    if let Ok(v) = env::var(name) {
        *target = parse_bool(&v);
    }
}
