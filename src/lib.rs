use apps::basic;
use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;

pub mod app;
pub mod apps;
pub mod buffer;
pub mod map;
pub mod render;
pub mod sim;
pub mod util;

#[derive(Debug, Clone, Parser)]
pub struct Cli {
    #[command(subcommand)]
    app: AppCli,
}

#[derive(Debug, Clone, Subcommand)]
pub enum AppCli {
    #[command(name = "basic")]
    Basic(basic::Cli),
}

impl Cli {
    pub fn run(self) -> Result<()> {
        self.app.run()
    }
}

impl AppCli {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Basic(basic) => basic.run(),
        }
    }
}
