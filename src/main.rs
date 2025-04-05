use color_eyre::eyre::Result;
use nephos::app::App;

fn main() -> Result<()> {
    App::new()?.run()
}
