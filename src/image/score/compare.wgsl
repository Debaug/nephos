@group(0) @binding(0) var tex_sampler: sampler;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(1) @binding(0) var rendered: texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(8, 8, 1) fn compare(
    @builtin(global_invocation_id) gid: vec3u, // 0..256
) {
    let coords = (vec2f(gid.xy) + 0.5) / 256.0;
    let source_value = textureSampleLevel(source, tex_sampler, coords, 0.0).x;
    let rendered_value = textureSampleLevel(rendered, tex_sampler, coords, 0.0).x;
    let result = select(0u, 1u, source_value == rendered_value);
    dst[gid.x * 256 + gid.y] = result;
}
