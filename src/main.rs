use std::fs;

mod config;
mod gpu;
mod visualization;
mod model;
mod preprocessing;
mod l1_bbox_gpu;

fn main() -> std::io::Result<()> {
    env_logger::init();
    let (device, _adapter, queue) = pollster::block_on(gpu::gpu_init());

    let l0_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer0.wgsl"));
    let l0_values = model::layer0_init(&device, l0_shader);
    let l1_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer1_bbox.wgsl"));
    let l1_pipelines = l1_bbox_gpu::build_l1_bbox_pipelines(&device, l1_shader);
    let mut l1_buffers: Option<l1_bbox_gpu::L1BboxBuffers> = None;
    let cfg = config::init();
    let images_dir = match cfg.images_dir.clone() {
        Some(v) => v,
        None => {
            eprintln!("IMAGES_DIR not set in config");
            return Ok(());
        }
    };
    let dir = std::env::current_dir()?.join(images_dir);
    let max_images = cfg.max_images;
    let test_image = cfg.test_image.clone();
    let mut processed: usize = 0;

    if let Some(path) = test_image {
        let (img_buffer, img_info) = match preprocessing::open_img_as_texture(&path, &device, &queue) {
            Ok(v) => v,
            Err(_e) => return Ok(()),
        };
        model::model(
            &device,
            &queue,
            img_buffer,
            img_info,
            &l0_values,
            &l1_pipelines,
            &mut l1_buffers,
            &path,
        );
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("png") {
                    if let Some(_file_name) = path.file_name() {
                        let (img_buffer, img_info) = match preprocessing::open_img_as_texture(&path, &device, &queue) {
                            Ok(v) => v,
                            Err(_) => {
                                continue;
                            }
                        };
                        model::model(
                            &device,
                            &queue,
                            img_buffer,
                            img_info,
                            &l0_values,
                            &l1_pipelines,
                            &mut l1_buffers,
                            &path,
                        );
                        processed += 1;
                        if processed >= max_images {
                            break;
                        }
                    }
                }
            }
        }
        if processed >= max_images {
            break;
        }
    }

    Ok(())
}
