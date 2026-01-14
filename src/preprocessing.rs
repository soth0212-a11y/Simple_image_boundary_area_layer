use std::path::Path;

pub struct Imginfo {
    pub height: u32,
    pub width: u32,
}

pub fn open_img_as_texture(
    path: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(wgpu::TextureView, Imginfo), image::ImageError> {
    let img = image::open(path)?;
    let (width, height) = (img.width(), img.height());

    let img_info = Imginfo { height, width };

    let rgba = img.to_rgba8().into_raw(); // Vec<u8>, len = w*h*4

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("input image texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Uint, // WGSL: texture_2d<u32>
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[][..],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // 주의: view만 반환하면 texture가 drop될 위험이 있으니,
    // 실전에서는 texture를 함께 보관해야 합니다.
    // 최소 변경으로는 view + texture를 같이 반환하는 게 안전합니다.

    Ok((view, img_info))
}
