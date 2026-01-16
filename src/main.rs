use std::fs;

mod config;
mod gpu;
mod visualization;
mod model;
mod preprocessing;
mod rle_ccl_gpu;

fn main() -> std::io::Result<()> {
    env_logger::init();
    let (device, _adapter, queue) = pollster::block_on(gpu::gpu_init());

    let l0_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer0.wgsl"));
    let l0_values = model::layer0_init(&device, l0_shader);

    let l1_count_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer1_rle_count.wgsl"));
    let l1_emit_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer1_rle_emit.wgsl"));
    let l2_init_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2_ccl_init.wgsl"));
    let l2_union_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2_ccl_union.wgsl"));
    let l2_reduce_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2_ccl_reduce.wgsl"));
    let l2_emit_boxes_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2_ccl_emit_boxes.wgsl"));
    let rle_ccl_pipelines = rle_ccl_gpu::build_rle_ccl_pipelines(
        &device,
        l1_count_shader,
        l1_emit_shader,
        l2_init_shader,
        l2_union_shader,
        l2_reduce_shader,
        l2_emit_boxes_shader,
    );
    let mut rle_ccl_buffers: Option<rle_ccl_gpu::RleCclBuffers> = None;

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
            &rle_ccl_pipelines,
            &mut rle_ccl_buffers,
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
                            &rle_ccl_pipelines,
                            &mut rle_ccl_buffers,
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
