@group(0) @binding(0) var<storage> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64) fn reduce(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let start_idx = gid.x * 2;
    dst[gid.x] = src[start_idx] + src[start_idx + 1];
}
