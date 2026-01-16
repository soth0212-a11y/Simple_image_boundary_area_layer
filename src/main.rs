use std::fs;

mod config;
mod gpu;
mod visualization;
mod model;
mod preprocessing;
mod l3_gpu;
mod l4_gpu;
mod l5;
#[cfg(test)]
mod l3;

fn main() -> std::io::Result<()> {
    env_logger::init();
    let (device, _adapter, queue) = pollster::block_on(gpu::gpu_init());
    

    let l0_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer0.wgsl"));
    let l1_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer1.wgsl"));
    let l0_values = model::layer0_init(&device, l0_shader);
    let l1_values = model::layer1_init(&device, l1_shader);
    let l2_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2.wgsl"));
    let l2_values = model::layer2_init(&device, l2_shader);
    let l3_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer3_stride2_conn8.wgsl"));
    let l3_pipelines = l3_gpu::build_l3_pipelines(&device, l3_shader);
    let mut l3_buffers: Option<l3_gpu::L3Buffers> = None;
    let l4_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer4_channels.wgsl"));
    let l4_pipelines = l4_gpu::build_l4_channels_pipelines(&device, l4_shader);
    let mut l4_buffers: Option<l4_gpu::L4ChannelsBuffers> = None;
    let l5_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer5_threshold_refine.wgsl"));
    let l5_pipelines = l5::build_l5_pipelines(&device, l5_shader);
    let mut l5_buffers: Option<l5::L5Buffers> = None;
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
            [&l0_values, &l1_values],
            &l2_values,
            &l3_pipelines,
            &mut l3_buffers,
            &l4_pipelines,
            &mut l4_buffers,
            &l5_pipelines,
            &mut l5_buffers,
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
                                // eprintln!("skip {:?}: {}", path, e);
                                continue; //if get a error skip
                            }
                        };           
                        // print!("height : {}, width : {}", img_info.height, img_info.width);
                        model::model(
                            &device,
                            &queue,
                            img_buffer,
                            img_info,
                            [&l0_values, &l1_values],
                            &l2_values,
                            &l3_pipelines,
                            &mut l3_buffers,
                            &l4_pipelines,
                            &mut l4_buffers,
                            &l5_pipelines,
                            &mut l5_buffers,
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
