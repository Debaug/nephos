use std::time::Duration;

use color_eyre::eyre::Result;
use nephos::{apps::basic, map::Sierpinski};

fn main() -> Result<()> {
    basic::run(Sierpinski, 1000, Duration::from_millis(200))
}
