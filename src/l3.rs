pub fn cpu_stride2_pool(in_active_mask: &[u32], in_w: usize, in_h: usize) -> (Vec<u32>, usize, usize) {
    let out_w = (in_w + 1) / 2;
    let out_h = (in_h + 1) / 2;
    let mut s_active = vec![0u32; out_w * out_h];
    for sy in 0..out_h {
        for sx in 0..out_w {
            let x0 = sx * 2;
            let y0 = sy * 2;
            let mut a = 0u32;
            for oy in 0..2 {
                let iy = y0 + oy;
                if iy >= in_h {
                    continue;
                }
                for ox in 0..2 {
                    let ix = x0 + ox;
                    if ix >= in_w {
                        continue;
                    }
                    let idx = iy * in_w + ix;
                    if idx >= in_active_mask.len() {
                        continue;
                    }
                    let v = (in_active_mask[idx] >> 1) & 1;
                    a |= v;
                }
            }
            s_active[sy * out_w + sx] = a & 1;
        }
    }
    (s_active, out_w, out_h)
}

pub fn cpu_conn8(s_active: &[u32], out_w: usize, out_h: usize) -> Vec<u32> {
    let mut conn8 = vec![0u32; out_w * out_h];
    for y in 0..out_h {
        for x in 0..out_w {
            let idx = y * out_w + x;
            if idx >= s_active.len() {
                continue;
            }
            if (s_active[idx] & 1) == 0 {
                conn8[idx] = 0;
                continue;
            }
            let mut mask = 0u32;
            let n = if y > 0 { s_active[(y - 1) * out_w + x] & 1 } else { 0 };
            let e = if x + 1 < out_w { s_active[y * out_w + (x + 1)] & 1 } else { 0 };
            let s = if y + 1 < out_h { s_active[(y + 1) * out_w + x] & 1 } else { 0 };
            let w = if x > 0 { s_active[y * out_w + (x - 1)] & 1 } else { 0 };
            let ne = if x + 1 < out_w && y > 0 { s_active[(y - 1) * out_w + (x + 1)] & 1 } else { 0 };
            let se = if x + 1 < out_w && y + 1 < out_h { s_active[(y + 1) * out_w + (x + 1)] & 1 } else { 0 };
            let sw = if x > 0 && y + 1 < out_h { s_active[(y + 1) * out_w + (x - 1)] & 1 } else { 0 };
            let nw = if x > 0 && y > 0 { s_active[(y - 1) * out_w + (x - 1)] & 1 } else { 0 };
            if n != 0 { mask |= 1 << 0; }
            if ne != 0 { mask |= 1 << 1; }
            if e != 0 { mask |= 1 << 2; }
            if se != 0 { mask |= 1 << 3; }
            if s != 0 { mask |= 1 << 4; }
            if sw != 0 { mask |= 1 << 5; }
            if w != 0 { mask |= 1 << 6; }
            if nw != 0 { mask |= 1 << 7; }
            conn8[idx] = mask;
        }
    }
    conn8
}

#[cfg(test)]
mod tests {
    use super::{cpu_conn8, cpu_stride2_pool};
    use crate::gpu;
    use crate::l3_gpu;
    use wgpu::util::DeviceExt;

    fn readback_u32(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, byte_len: u64) -> Vec<u32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("l3_test_readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    fn run_gpu(in_mask: &[u32], in_w: u32, in_h: u32) -> (Vec<u32>, Vec<u32>, usize, usize) {
        let (device, _adapter, queue) = pollster::block_on(gpu::gpu_init());
        let shader = device.create_shader_module(wgpu::include_wgsl!("./wgsl/layer3_stride2_conn8.wgsl"));
        let pipelines = l3_gpu::build_l3_pipelines(&device, shader);
        let out_w = (in_w + 1) / 2;
        let out_h = (in_h + 1) / 2;
        let bufs = l3_gpu::ensure_l3_buffers(&device, in_w, in_h, out_w, out_h);
        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("l3_test_in_active"),
            contents: bytemuck::cast_slice(in_mask),
            usage: wgpu::BufferUsages::STORAGE,
        });
        l3_gpu::dispatch_l3(&device, &queue, &pipelines, &bufs, &in_buf);
        let out_len = (out_w * out_h) as u64 * 4;
        let s_active = readback_u32(&device, &queue, &bufs.s_active, out_len);
        let conn8 = readback_u32(&device, &queue, &bufs.conn8, out_len);
        (s_active, conn8, out_w as usize, out_h as usize)
    }

    #[test]
    fn l3_stride2_pool_and_conn8_4x4() {
        let in_w = 4usize;
        let in_h = 4usize;
        let mut in_mask = vec![0u32; in_w * in_h];
        in_mask[0] = 1u32 << 1;
        in_mask[1 * in_w + 1] = 1u32 << 1;
        in_mask[3 * in_w + 3] = 1u32 << 1;

        let (cpu_s, out_w, out_h) = cpu_stride2_pool(&in_mask, in_w, in_h);
        let cpu_conn = cpu_conn8(&cpu_s, out_w, out_h);
        let (gpu_s, gpu_conn, gw, gh) = run_gpu(&in_mask, in_w as u32, in_h as u32);

        assert_eq!((out_w, out_h), (gw, gh));
        assert_eq!(cpu_s, gpu_s);
        assert_eq!(cpu_conn, gpu_conn);
    }

    #[test]
    fn l3_stride2_pool_and_conn8_odd_dims() {
        let in_w = 5usize;
        let in_h = 3usize;
        let mut in_mask = vec![0u32; in_w * in_h];
        in_mask[0] = 1u32 << 1;
        in_mask[2] = 1u32 << 1;
        in_mask[1 * in_w + 4] = 1u32 << 1;
        in_mask[2 * in_w + 1] = 1u32 << 1;

        let (cpu_s, out_w, out_h) = cpu_stride2_pool(&in_mask, in_w, in_h);
        let cpu_conn = cpu_conn8(&cpu_s, out_w, out_h);
        let (gpu_s, gpu_conn, gw, gh) = run_gpu(&in_mask, in_w as u32, in_h as u32);

        assert_eq!((out_w, out_h), (gw, gh));
        assert_eq!(cpu_s, gpu_s);
        assert_eq!(cpu_conn, gpu_conn);
    }
}
