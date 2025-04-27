use std::sync::{mpsc, Arc};

use clap::Parser;
use color_eyre::eyre::{Ok, Result};
use futures::future::BoxFuture;
use glam::Affine2;
use image::ImageReader;
use wgpu::{Features, Limits, SurfaceConfiguration};
use winit::{dpi::LogicalSize, event::WindowEvent, window::WindowAttributes};

use crate::{
    app::{self, Context, LocalAppController, Run},
    image::Evolver,
    map::Map,
    render::{Camera, Renderer},
    util::SyncingFuture,
};

#[derive(Debug, Clone, Parser)]
pub struct Cli {
    // #[arg(short)]
    // pub n_points: usize,

    // #[arg(short, long = "delta", default_value_t = 250)]
    // pub delta_time_ms: u64,

    // #[arg(short, long, requires = "n_gens")]
    // pub out: Option<PathBuf>,

    // #[arg(short = 'g', long, requires = "out")]
    // pub n_gens: Option<usize>,
}

impl Cli {
    pub fn run(self) -> Result<()> {
        Run::new(AppBuilder {
            generations: 10000,
            maps_per_set: 6,
            elite_len: 3,
            depth: 15,
            n_children: 20,
            n_points: 50000,
            mutation_strength: 1.0,
            mutation_damping: 0.02,
        })
        .with_window_attributes(
            WindowAttributes::default().with_inner_size(LogicalSize::new(600, 600)),
        )
        .with_features(Features::PUSH_CONSTANTS)
        .with_limits(Limits {
            max_push_constant_size: 8,
            ..Default::default()
        })
        .run()
    }
}

struct AppBuilder {
    generations: usize,
    maps_per_set: usize,
    elite_len: usize,
    depth: usize,
    n_children: usize,
    n_points: usize,
    mutation_strength: f32,
    mutation_damping: f32,
}

struct App {
    evolver: Arc<Evolver>,
    renderer: Renderer,
    camera: Camera,
    best_map_set: mpsc::Receiver<Vec<Map>>,
}

impl app::AppBuilder for AppBuilder {
    type App = App;

    fn build(
        self,
        surface_configuration: &SurfaceConfiguration,
        context: Context,
    ) -> BoxFuture<'static, Result<Self::App>> {
        env_logger::init();
        let context = context.into_static();

        let image = ImageReader::open("assets/source.png")
            .expect("failed to read source image")
            .decode()
            .expect("failed to decode source image")
            .to_luma8();

        let (tx, rx) = mpsc::channel();

        let renderer = Renderer::new(context.borrow(), surface_configuration.format);
        let camera = Camera::new(Affine2::IDENTITY, context.borrow());

        let evolver = Arc::new(
            Evolver::new(
                &image,
                self.maps_per_set,
                self.elite_len,
                self.depth,
                self.n_children,
                self.n_points,
                self.mutation_strength,
                self.mutation_damping,
                context.borrow(),
            )
            .expect("failed to create evolver"),
        );

        let evolver2 = evolver.clone();
        let generations = self.generations;
        let context2 = context.to_static();
        context.runtime().spawn(async move {
            evolver2
                .evolve(generations, context2.borrow().to_static())
                .await;
            let best_map_set = evolver2.get_best_map_set(context2).await;
            dbg!(&best_map_set);
            tx.send(best_map_set).expect("failed to send message");
        });

        Box::pin(async move {
            Ok(App {
                evolver,
                renderer,
                camera,
                best_map_set: rx,
            })
        })
    }
}

impl app::App for App {
    fn event(
        &mut self,
        event: winit::event::WindowEvent,
        _context: app::Context,
        controller: LocalAppController,
    ) {
        if event == WindowEvent::CloseRequested {
            controller.exit();
        }
    }

    fn render(&mut self, target: &wgpu::SurfaceTexture, context: app::Context) -> Result<()> {
        let points = self.evolver.get_some_points();
        self.renderer
            .render(points, &self.camera, target, context)
            .ignore();
        Ok(())
    }
}
