@group(0) @binding(0) var<uniform> inverse_camera: mat3x3<f32>;

@vertex fn vertex(@location(0) point: vec2<f32>) -> @builtin(position) vec4<f32> {
    let clip_point = (inverse_camera * vec3<f32>(point, 1.0)).xy;
    return vec4<f32>(clip_point, 0.0, 1.0);
}

@fragment fn fragment(@builtin(position) point: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
