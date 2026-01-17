use std::fs;

mod config;
mod gpu;
mod l2_v3_gpu;
mod layer3_tile32;
mod visualization_l3_tile32;
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
    let l2_clear_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2_v3_clear.wgsl"));
    let l2_reduce_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2_v3_group_reduce.wgsl"));
    let l2_expand_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2_v3_expand.wgsl"));
    let l2_emit_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2_v3_emit.wgsl"));
    let l3_clear_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer3_tile32_clear.wgsl"));
    let l3_accum_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer3_tile32_accumulate.wgsl"));
    let l3_emit_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer3_tile32_emit.wgsl"));

    let l1_rle_pipelines = rle_ccl_gpu::build_l1_rle_pipelines(
        &device,
        l1_count_shader,
        l1_emit_shader,
    );
    let l2_v3_pipelines = l2_v3_gpu::build_l2_v3_pipelines(
        &device,
        l2_clear_shader,
        l2_reduce_shader,
        l2_expand_shader,
        l2_emit_shader,
    );
    let l3_tile32_pipelines = layer3_tile32::build_l3_tile32_pipelines(&device, l3_clear_shader, l3_accum_shader, l3_emit_shader);
    let mut l1_rle_buffers: Option<rle_ccl_gpu::L1RleBuffers> = None;
    let mut l2_v3_buffers: Option<l2_v3_gpu::L2v3Buffers> = None;
    let mut l3_tile32_buffers: Option<layer3_tile32::L3Tile32Buffers> = None;

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
            &l1_rle_pipelines,
            &mut l1_rle_buffers,
            &l2_v3_pipelines,
            &mut l2_v3_buffers,
            &l3_tile32_pipelines,
            &mut l3_tile32_buffers,
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
                &l1_rle_pipelines,
                &mut l1_rle_buffers,
                &l2_v3_pipelines,
                &mut l2_v3_buffers,
                &l3_tile32_pipelines,
                &mut l3_tile32_buffers,
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
