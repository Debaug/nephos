use std::f32;

use glam::{vec2, Affine2, Mat2, Vec2};

use crate::util::{mat2, Affine2Ext};

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

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub fn to_clip_transform(&self) -> Affine2 {
        let midpoint = 0.5 * (self.min + self.max);
        let scale = self.max - midpoint;
        Affine2::from_scale_angle_translation(scale, 0.0, midpoint)
    }
}

pub trait Maps {
    fn region(&self) -> Rect {
        Rect {
            min: Vec2::NEG_ONE,
            max: Vec2::ONE,
        }
    }
    fn maps(&self) -> Vec<Map>;
}

impl Maps for [Map] {
    fn maps(&self) -> Vec<Map> {
        self.to_vec()
    }
}

pub struct Sierpinski;

impl Maps for Sierpinski {
    fn maps(&self) -> Vec<Map> {
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
    fn maps(&self) -> Vec<Map> {
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
    fn maps(&self) -> Vec<Map> {
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
    fn maps(&self) -> Vec<Map> {
        // based on https://en.wikipedia.org/wiki/Barnsley_fern
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

    fn region(&self) -> Rect {
        Rect {
            min: vec2(-5.0, 0.0),
            max: vec2(5.0, 10.0),
        }
    }
}

pub struct SillySquare;

impl Maps for SillySquare {
    fn maps(&self) -> Vec<Map> {
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
    fn maps(&self) -> Vec<Map> {
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
    fn maps(&self) -> Vec<Map> {
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

pub struct Pipistrello;

impl Maps for Pipistrello {
    fn maps(&self) -> Vec<Map> {
        vec![
            Affine2::from_scale(vec2(0.25, 0.5))
                .with_center(vec2(0.0, -0.75))
                .into(),
            Affine2::from_scale_angle_translation(
                vec2(0.5, 0.5),
                f32::consts::FRAC_PI_3,
                Vec2::ZERO,
            )
            .with_center(vec2(-0.5, -0.5))
            .into(),
            Affine2::from_scale_angle_translation(
                vec2(0.67, 0.9),
                f32::consts::FRAC_PI_4,
                Vec2::ZERO,
            )
            .with_center(vec2(0.25, 0.25))
            .into(),
        ]
    }
}

pub struct Tunnel;

impl Maps for Tunnel {
    fn maps(&self) -> Vec<Map> {
        let mid = Affine2::from_mat2(mat2(
            0.5, -0.5, //
            0.5, 0.5,
        ));
        let side = Affine2::from_scale_angle_translation(vec2(0.25, 0.5), 0.0, vec2(0.75, 0.0));
        vec![
            mid.into(),
            side.into(),
            (Affine2::from_angle(f32::consts::FRAC_PI_2) * side).into(),
            (Affine2::from_angle(f32::consts::PI) * side).into(),
            (Affine2::from_angle(-f32::consts::FRAC_PI_2) * side).into(),
        ]
    }
}

pub struct Weed;

impl Maps for Weed {
    fn maps(&self) -> Vec<Map> {
        let stem = Affine2::from_mat2_translation(
            mat2(
                0.0, 0.0, //
                0.0, 0.05,
            ),
            vec2(0.0, -0.95),
        );

        let up = Affine2::from_scale_angle_translation(vec2(0.9, 0.9), 0.0, vec2(0.0, 0.55));

        let left = Affine2::from_scale_angle_translation(
            vec2(0.5, 0.5),
            f32::consts::FRAC_PI_3,
            vec2(0.0, -0.9),
        ) * Affine2::from_translation(vec2(0.0, 0.5));

        let right = Affine2::from_mat2(mat2(
            -1.0, 0.0, //
            0.0, 1.0,
        )) * left;

        vec![stem.into(), up.into(), left.into(), right.into()]
    }
}

// Made with Tiziano Piserchia
pub struct FleurAstrale;

impl Maps for FleurAstrale {
    fn maps(&self) -> Vec<Map> {
        let corner = Affine2 {
            matrix2: Mat2::from_diagonal(Vec2::splat(0.3)),
            translation: vec2(0.67, 0.67),
        };
        let corner2 = Affine2::from_mat2(mat2(
            -1.0, 0.0, //
            0.0, 1.0,
        )) * corner;
        vec![
            Affine2::from_scale_angle_translation(
                vec2(0.67, 0.67),
                f32::consts::FRAC_PI_4,
                Vec2::ZERO,
            )
            .into(),
            corner.into(),
            corner2.into(),
            (Affine2::from_mat2(-Mat2::IDENTITY) * corner).into(),
            (Affine2::from_mat2(-Mat2::IDENTITY) * corner2).into(),
        ]
    }
}

pub struct Pentagon;

impl Maps for Pentagon {
    fn maps(&self) -> Vec<Map> {
        // let tr1 = Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, vec2(0.0, 0.5));
        (0..5)
            .map(|i| {
                let angle = f32::consts::TAU / 5.0 * i as f32;
                let center = Mat2::from_angle(angle) * vec2(0.0, 0.80);
                Affine2::from_scale(vec2(0.5, 0.5))
                    .with_center(center)
                    .into()
            })
            .collect()
    }
}
