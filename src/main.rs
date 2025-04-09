use std::{iter, time::Duration};

use color_eyre::eyre::Result;
use glam::Vec2;
use nephos::{app::App, map::*, sim::Point};
use rand::Rng;

fn main() -> Result<()> {
    env_logger::init();

    let mut rng = rand::rng();
    let points: Vec<_> = iter::repeat_with(|| Point {
        position: Vec2::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)),
    })
    .take(200000)
    .collect();

    App::new(Duration::from_millis(200), points, Sierpinski)?.run()
}
