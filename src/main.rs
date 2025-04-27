use clap::Parser;
use color_eyre::eyre::Result;
use nephos::apps::basic;
fn main() -> Result<()> {
    basic::Cli::try_parse()?.run()
    // nephos::apps::fit::Cli::try_parse()?.run()
}
