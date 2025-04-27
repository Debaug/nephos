//!include src/image/common.wgsl

@group(0) @binding(0) var<storage> maps: array<Affine>;
@group(1) @binding(0) var<storage, read_write> points: array<vec2f>;
var<push_constant> map_set_range: array<u32, 2>;

@compute @workgroup_size(1) fn step_sim(
    @builtin(global_invocation_id) id: vec3u,
) {
    let point = points[id.x];
    let map = maps[map_set_range[0] + hash_point(point) % map_set_range[1]].computed;
    points[id.x] = (map * vec3(point, 1.0)).xy;
}

fn hash_point(point: vec2f) -> u32 {
    return bitcast<u32>(point.x + point.y);
}
