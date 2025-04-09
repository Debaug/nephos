use glam::{Mat2, Vec2};

// matrix of the form
// m00 m01
// m10 m11
pub fn mat2(m00: f32, m01: f32, m10: f32, m11: f32) -> Mat2 {
    Mat2::from_cols(Vec2::new(m00, m10), Vec2::new(m01, m11))
}
