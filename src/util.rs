use std::future::Future;

use bytemuck::{Pod, Zeroable};
use glam::{Affine2, Mat2, Mat3, Vec2, Vec4};
use wgpu_async::WgpuFuture;

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

/// A future that doesn't do any work upon polling, but rather serves to signal that a computation is done.
pub trait SyncingFuture: Future<Output = ()> + 'static {
    fn ignore(self);
}

impl SyncingFuture for WgpuFuture<()> {
    fn ignore(self) {
        drop(self);
    }
}

pub trait Affine2Ext {
    fn with_center(self, center: Vec2) -> Self;
}

impl Affine2Ext for Affine2 {
    fn with_center(self, center: Vec2) -> Self {
        Affine2::from_translation(center) * self * Affine2::from_translation(-center)
    }
}
