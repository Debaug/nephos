use clap::Parser;
use color_eyre::eyre::Result;
use nephos::Cli;

fn main() -> Result<()> {
    Cli::try_parse()?.run()
}
