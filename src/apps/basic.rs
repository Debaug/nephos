use std::{
    iter,
    sync::{
        mpsc::{self, TryRecvError},
        Arc,
    },
    time::Duration,
};

use color_eyre::eyre::{Ok, Result};
use futures::future::BoxFuture;
use glam::Vec2;
use rand::Rng;
use wgpu::{BufferUsages, SurfaceConfiguration};
use winit::{dpi::LogicalSize, event::WindowEvent, window::WindowAttributes};

use crate::{
    app::{self, Context, LocalAppController, Run},
    buffer::Buffer,
    map::{Map, Maps},
    render::Renderer,
    sim::{Point, Simulation},
};

pub fn run(maps: impl Maps, n_points: usize, delta_time: Duration) -> Result<()> {
    Run::new(AppBuilder {
        maps: maps.into_maps(),
        n_points,
        delta_time,
    })
    .with_window_attributes(WindowAttributes::default().with_inner_size(LogicalSize::new(600, 600)))
    .run()
}

struct AppBuilder {
    maps: Vec<Map>,
    n_points: usize,
    delta_time: Duration,
}

struct App {
    simulation: Arc<Simulation<Buffer<Point>>>,
    renderer: Renderer,
    stop_simulation_tx: mpsc::Sender<()>,
}

impl app::AppBuilder for AppBuilder {
    type App = App;

    fn build(
        self,
        surface_configuration: &SurfaceConfiguration,
        context: Context,
    ) -> BoxFuture<'static, Result<Self::App>> {
        env_logger::init();

        let mut rng = rand::rng();
        let points: Vec<_> = iter::repeat_with(|| Point {
            position: Vec2::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)),
        })
        .take(self.n_points)
        .collect();

        let point_buffer = Buffer::new(
            &points,
            Some("Points"),
            BufferUsages::STORAGE | BufferUsages::VERTEX,
            context.borrow(),
        );

        let simulation = Arc::new(Simulation::new(point_buffer, &self.maps, context.borrow()));
        let renderer = Renderer::new(context.borrow(), surface_configuration.format);

        let (stop_simulation_tx, stop_simulation_rx) = mpsc::channel();
        let simulation2 = simulation.clone();
        let context2 = context.to_static();
        context.borrow().runtime().spawn(async move {
            let mut interval = tokio::time::interval(self.delta_time);
            while stop_simulation_rx.try_recv() == Err(TryRecvError::Empty) {
                interval.tick().await;
                simulation2.step(context2.borrow()).await;
            }
        });

        let app = App {
            simulation,
            renderer,
            stop_simulation_tx,
        };

        Box::pin(async move { Ok(app) })
    }
}

impl App {}

impl Drop for App {
    fn drop(&mut self) {
        self.stop_simulation_tx
            .send(())
            .expect("failed to stop simulation");
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
        drop(
            self.renderer
                .render(self.simulation.points(), target, context.borrow()),
        );

        Ok(())
    }
}
