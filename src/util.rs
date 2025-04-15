use bytemuck::{Pod, Zeroable};
use glam::{Mat2, Mat3, Vec2, Vec4};

// matrix of the form
// m00 m01
// m10 m11
pub fn mat2(m00: f32, m01: f32, m10: f32, m11: f32) -> Mat2 {
    Mat2::from_cols(Vec2::new(m00, m10), Vec2::new(m01, m11))
}

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct WgpuMat3x3([Vec4; 3]);

impl From<Mat3> for WgpuMat3x3 {
    fn from(mat: Mat3) -> Self {
        WgpuMat3x3([
            mat.col(0).extend(0.0),
            mat.col(1).extend(0.0),
            mat.col(2).extend(0.0),
        ])
    }
}

impl From<WgpuMat3x3> for Mat3 {
    fn from(mat: WgpuMat3x3) -> Self {
        let [x, y, z] = mat.0;
        Mat3::from_cols(x.truncate(), y.truncate(), z.truncate())
    }
}
