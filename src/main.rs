use std::fs;
use std::time::Instant;

mod config;
mod gpu;
mod visualization;
mod model;
mod preprocessing;
mod l3_edge_gpu;
mod l4_gpu;

fn main() -> std::io::Result<()> {
    env_logger::init();
    let (device, adapter, queue) = pollster::block_on(gpu::gpu_init());
    

    let l0_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer0.wgsl"));
    let l1_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer1.wgsl"));
    let l0_values = model::layer0_init(&device, l0_shader);
    let l1_values = model::layer1_init(&device, l1_shader);
    let l2_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer2.wgsl"));
    let l2_values = model::layer2_init(&device, l2_shader);
    let layer3_pass1_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer3_edge_pass1.wgsl"));
    let layer3_pass2a_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer3_edge_pass2a.wgsl"));
    let layer3_pass2b_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer3_edge_pass2b.wgsl"));
    let l3_edge_pipelines = l3_edge_gpu::build_l3_edge_pipelines(&device, layer3_pass1_shader, layer3_pass2a_shader, layer3_pass2b_shader);
    let mut l3_edge_buffers: Option<l3_edge_gpu::L3EdgeGpuBuffers> = None;
    let layer4_expand_shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer4_expand_once.wgsl"));
    let l4_pipelines = l4_gpu::build_l4_expand_pipelines(&device, layer4_expand_shader);
    let mut l4_buffers: Option<l4_gpu::L4ExpandBuffers> = None;

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
        let start = Instant::now();
        model::model(
            &device,
            &queue,
            img_buffer,
            img_info,
            [&l0_values, &l1_values],
            &l2_values,
            &l3_edge_pipelines,
            &mut l3_edge_buffers,
            &l4_pipelines,
            &mut l4_buffers,
            &path,
        );
        let _duration = start.elapsed();
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
                        let start = Instant::now(); // start time record
                        model::model(
                            &device,
                            &queue,
                            img_buffer,
                            img_info,
                            [&l0_values, &l1_values],
                            &l2_values,
                            &l3_edge_pipelines,
                            &mut l3_edge_buffers,
                            &l4_pipelines,
                            &mut l4_buffers,
                            &path,
                        );
                        let _ = start.elapsed(); // time check
                        // println!("without image load total run time : {:?} \n", duration);
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
