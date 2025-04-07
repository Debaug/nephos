@group(0) @binding(0) var<storage, read_write> points: array<vec2<f32>>;
@group(0) @binding(1) var<storage> maps: array<mat3x3<f32>>;

@compute @workgroup_size(1) fn kernel(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let point = points[id.x];
    let map = maps[hash(point) % arrayLength(&maps)];
    points[id.x] = (map * vec3(point, 1.0)).xy;
}

fn hash(point: vec2<f32>) -> u32 {
    return bitcast<u32>(point.x * point.y);
}
