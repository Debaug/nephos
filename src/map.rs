use std::f32;

use glam::{vec2, Affine2, Vec2};

use crate::util::mat2;

#[derive(Debug, Clone, Copy)]
pub struct Map {
    pub map: Affine2,
    pub probability_weight: f32,
}

impl From<Affine2> for Map {
    fn from(map: Affine2) -> Self {
        Self {
            map,
            probability_weight: 1.0,
        }
    }
}

pub trait Maps {
    fn into_maps(self) -> Vec<Map>;
}

impl Maps for Vec<Map> {
    fn into_maps(self) -> Vec<Map> {
        self
    }
}

pub struct Sierpinski;

impl Maps for Sierpinski {
    fn into_maps(self) -> Vec<Map> {
        vec![
            Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::new(0.0, 0.5))
                .into(),
            Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::splat(-0.5)).into(),
            Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::new(0.5, -0.5))
                .into(),
        ]
    }
}
pub struct Yang;

impl Maps for Yang {
    fn into_maps(self) -> Vec<Map> {
        vec![
            Affine2::from_scale_angle_translation(Vec2::splat(0.9), -0.5, Vec2::new(0.0, 0.1))
                .into(),
            Affine2::from_scale_angle_translation(Vec2::splat(0.9), 0.2, Vec2::new(0.0, 0.1))
                .into(),
        ]
    }
}

pub struct Polygon {
    pub n: u32,
}

impl Maps for Polygon {
    fn into_maps(self) -> Vec<Map> {
        // based on https://en.wikipedia.org/wiki/Chaos_game
        let _r = match self.n % 4 {
            0 => 1.0 / (1.0 + f32::tan(f32::consts::PI / self.n as f32)),
            1 | 3 => 1.0 / (1.0 + 2.0 * f32::sin(f32::consts::PI / (2 * self.n) as f32)),
            2 => 1.0 / (1.0 + f32::sin(f32::consts::PI / self.n as f32)),
            _ => unreachable!(),
        };

        todo!()
    }
}

pub struct Barnsley;

impl Maps for Barnsley {
    fn into_maps(self) -> Vec<Map> {
        vec![
            Map {
                map: Affine2::from_mat2(mat2(
                    0.0, 0.0, //
                    0.0, 0.16,
                )),
                probability_weight: 0.01,
            },
            Map {
                map: Affine2::from_mat2_translation(
                    mat2(
                        0.85, 0.04, //
                        -0.04, 0.85,
                    ),
                    vec2(0.0, 1.6),
                ),
                probability_weight: 0.85,
            },
            Map {
                map: Affine2::from_mat2_translation(
                    mat2(
                        0.2, -0.26, //
                        0.23, 0.22,
                    ),
                    vec2(0.0, 1.6),
                ),
                probability_weight: 0.07,
            },
            Map {
                map: Affine2::from_mat2_translation(
                    mat2(
                        -0.15, 0.28, //
                        0.26, 0.24,
                    ),
                    vec2(0.0, 0.44),
                ),
                probability_weight: 0.07,
            },
        ]
    }
}

pub struct SillySquare;

impl Maps for SillySquare {
    fn into_maps(self) -> Vec<Map> {
        let next = Affine2::from_scale_angle_translation(
            Vec2::splat(0.5),
            f32::consts::FRAC_PI_6,
            Vec2::new(-1.0, 1.0),
        );
        vec![next.into(), next.inverse().into(), Affine2::IDENTITY.into()]
    }
}

pub struct Patrick;

impl Maps for Patrick {
    fn into_maps(self) -> Vec<Map> {
        vec![
            Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::new(0.0, 0.5))
                .into(),
            Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::new(0.0, -0.5))
                .into(),
            Affine2::from_angle(f32::consts::FRAC_PI_2).into(),
        ]
    }
}

pub struct Disc;

impl Maps for Disc {
    fn into_maps(self) -> Vec<Map> {
        vec![
            Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::new(0.0, 0.5))
                .into(),
            Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::new(0.0, -0.5))
                .into(),
            Affine2::from_angle(3.88).into(),
            Affine2::from_angle(-2.0).into(),
        ]
    }
}
