use std::{iter, time::Duration};

use color_eyre::eyre::Result;
use glam::{Affine2, Vec2};
use nephos::{app::App, sim::Point};
use rand::Rng;

fn main() -> Result<()> {
    env_logger::init();

    let mut rng = rand::rng();
    let points: Vec<_> = iter::repeat_with(|| Point {
        position: Vec2::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)),
    })
    .take(60000)
    .collect();

    let maps = vec![
        Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::new(0.0, 0.36)),
        Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::splat(-0.5)),
        Affine2::from_scale_angle_translation(Vec2::splat(0.5), 0.0, Vec2::new(0.5, -0.5)),
    ];

    App::new(Duration::from_millis(200), points, maps)?.run()
}
