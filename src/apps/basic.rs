use std::{
    fs::{File, OpenOptions},
    iter,
    path::PathBuf,
    sync::{
        mpsc::{self, TryRecvError},
        Arc,
    },
    time::Duration,
};

use clap::Parser;
use color_eyre::eyre::{Ok, Result};
use futures::future::BoxFuture;
use glam::Vec2;
use rand::Rng;
use wgpu::{
    BufferUsages, CommandEncoderDescriptor, Extent3d, Origin3d, SurfaceConfiguration,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureDescriptor,
    TextureUsages, TextureView, TextureViewDescriptor,
};
use winit::{dpi::LogicalSize, event::WindowEvent, window::WindowAttributes};

use crate::{
    app::{self, Context, LocalAppController, Run},
    buffer::Buffer,
    map::{Map, Maps, Sierpinski},
    render::Renderer,
    sim::{Point, Simulation},
};

#[derive(Debug, Clone, Parser)]
pub struct Cli {
    #[arg(short)]
    pub n_points: usize,

    #[arg(short, long = "delta", default_value_t = 250)]
    pub delta_time_ms: u64,

    #[arg(short, long, requires = "n_gens")]
    pub out: Option<PathBuf>,

    #[arg(short = 'g', long, requires = "out")]
    pub n_gens: Option<usize>,
}

impl Cli {
    pub fn run(self) -> Result<()> {
        let record = self
            .out
            .map(|out| -> Result<_> {
                let file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(out)?;

                let n_gens = self.n_gens.unwrap();

                let width = 512;
                let height = 512;

                let encoder = gif::Encoder::new(file, width, height, &[])?;

                Ok(RecordConfig {
                    encoder,
                    n_gens,
                    width,
                    height,
                })
            })
            .transpose()?;

        Run::new(AppBuilder {
            maps: Sierpinski.into_maps(),
            n_points: self.n_points,
            delta_time: Duration::from_millis(self.delta_time_ms),
            record,
        })
        .with_window_attributes(
            WindowAttributes::default().with_inner_size(LogicalSize::new(600, 600)),
        )
        .run()
    }
}

struct AppBuilder {
    maps: Vec<Map>,
    n_points: usize,
    delta_time: Duration,
    record: Option<RecordConfig>,
}

struct RecordConfig {
    encoder: gif::Encoder<File>,
    n_gens: usize,
    width: u16,
    height: u16,
}

struct App {
    simulation: Arc<Simulation<Buffer<Point>>>,
    renderer: Renderer,
    stop_simulation_tx: mpsc::Sender<()>,
}

struct Record {
    encoder: gif::Encoder<File>,
    n_gens: usize,
    width: u16,
    height: u16,
    texture: Texture,
    texture_view: TextureView,
    buffer: Buffer<u8>,
    renderer: Renderer,
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

        let mut record = self.record.map(
            |RecordConfig {
                 encoder,
                 n_gens,
                 width,
                 height,
             }| {
                let texture = context.device().create_texture(&TextureDescriptor {
                    label: Some("Simulation Texture"),
                    size: Extent3d {
                        width: width.into(),
                        height: height.into(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                    view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                });

                let texture_view = texture.create_view(&TextureViewDescriptor {
                    label: Some("Simulation Texture View"),
                    ..Default::default()
                });

                let buffer = Buffer::new(
                    &vec![0; 512 * 512 * 4],
                    Some("Simulation Texture Buffer"),
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    context.borrow(),
                );

                let renderer = Renderer::new(context.borrow(), wgpu::TextureFormat::Rgba8Unorm);

                Record {
                    encoder,
                    n_gens,
                    width,
                    height,
                    texture,
                    texture_view,
                    buffer,
                    renderer,
                }
            },
        );

        let (stop_simulation_tx, stop_simulation_rx) = mpsc::channel();
        let simulation2 = simulation.clone();
        let context2 = context.to_static();

        context.borrow().runtime().spawn(async move {
            let context = context2;
            let simulation = simulation2;

            if let Some(record) = &mut record {
                record
                    .encoder
                    .set_repeat(gif::Repeat::Infinite)
                    .expect("failed to set repeating behavior of GIF");
            }

            let mut interval = tokio::time::interval(self.delta_time);
            let mut gen_iter = record.as_ref().map(|record| 0..record.n_gens);
            while stop_simulation_rx.try_recv() == Err(TryRecvError::Empty)
                && gen_iter.as_mut().is_none_or(|gen| gen.next().is_some())
            {
                interval.tick().await;
                simulation.step(context.borrow()).await;

                let Some(record) = &mut record else {
                    continue;
                };

                drop(record.renderer.render(
                    simulation.points(),
                    &record.texture_view,
                    context.borrow(),
                ));

                let mut copy_encoder =
                    context
                        .device()
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("Texture to Buffer command encoder"),
                        });
                copy_encoder.copy_texture_to_buffer(
                    TexelCopyTextureInfo {
                        texture: &record.texture,
                        mip_level: 0,
                        aspect: wgpu::TextureAspect::All,
                        origin: Origin3d::ZERO,
                    },
                    TexelCopyBufferInfo {
                        buffer: &record.buffer,
                        layout: TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(512 * 4),
                            rows_per_image: None,
                        },
                    },
                    Extent3d {
                        width: 512,
                        height: 512,
                        depth_or_array_layers: 1,
                    },
                );
                context.queue().submit(iter::once(copy_encoder.finish()));

                let slice = record.buffer.slice(..);
                slice
                    .map_async(wgpu::MapMode::Read)
                    .await
                    .expect("failed to map buffer");
                let mut bytes = slice.get_mapped_range().to_vec();
                record.buffer.unmap();

                let mut frame = gif::Frame::from_rgba(record.width, record.height, &mut bytes);
                frame.delay = (self.delta_time.as_millis() / 10).try_into().unwrap();

                record
                    .encoder
                    .write_frame(&frame)
                    .expect("failed to encode GIF frame");
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
