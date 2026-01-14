@group(0) @binding(0) var<storage, read> block_boxes: array<u32>;
@group(0) @binding(1) var<storage, read> block_valid: array<u32>;
@group(0) @binding(2) var<storage, read_write> flat_boxes: array<u32>;
@group(0) @binding(3) var<storage, read_write> flat_count: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> label: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot: u32 = gid.x;
    if (slot >= arrayLength(&block_valid)) { return; }
    if ((block_valid[slot] & 1u) == 0u) { return; }

    let bbox_idx: u32 = slot * 2u;
    if (bbox_idx + 1u >= arrayLength(&block_boxes)) { return; }

    let out_idx: u32 = atomicAdd(&flat_count, 1u);
    let max_boxes: u32 = arrayLength(&flat_boxes) / 2u;
    if (out_idx >= max_boxes) { return; }

    flat_boxes[out_idx * 2u] = block_boxes[bbox_idx];
    flat_boxes[out_idx * 2u + 1u] = block_boxes[bbox_idx + 1u];
    atomicStore(&label[out_idx], out_idx);
}
