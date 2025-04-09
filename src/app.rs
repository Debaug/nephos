use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use color_eyre::eyre::Result;
use log::error;
use thiserror::Error;
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Queue,
    RequestAdapterOptions, Surface, SurfaceConfiguration, TextureUsages,
};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize, Size},
    event::WindowEvent,
    event_loop::{self, ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

use crate::{
    map::{Map, Maps},
    render::Renderer,
    sim::{Point, Simulation},
};

#[derive(Debug)]
pub struct App {
    event_loop: EventLoop<()>,
    inner: AppInner,
}

#[derive(Debug)]
struct AppInner {
    instance: Instance,
    window_app: Option<WindowApp>,
    delta_time: Duration,
    points: Vec<Point>,
    maps: Vec<Map>,
    last_tick: Instant,
}

#[derive(Debug)]
struct WindowApp {
    adapter: Adapter,
    device: Device,
    queue: Queue,
    window: Arc<Window>,
    surface_configuration: SurfaceConfiguration,
    surface: Surface<'static>,
    simulation: Simulation,
    renderer: Renderer,
}

#[derive(Debug, Clone, Copy)]
pub struct Context<'app> {
    pub instance: &'app Instance,
    pub adapter: &'app Adapter,
    pub device: &'app Device,
    pub queue: &'app Queue,
    pub window: &'app Window,
    pub surface: &'app Surface<'static>,
    pub surface_configuration: &'app SurfaceConfiguration,
}

impl App {
    pub fn new(delta_time: Duration, points: Vec<Point>, maps: impl Maps) -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(event_loop::ControlFlow::Poll);

        Ok(Self {
            event_loop,
            inner: AppInner {
                instance,
                window_app: None,
                delta_time,
                points,
                maps: maps.into_maps(),
                last_tick: Instant::now(),
            },
        })
    }

    pub fn run(mut self) -> Result<()> {
        self.event_loop.run_app(&mut self.inner)?;
        Ok(())
    }
}

impl AppInner {
    fn tick(&mut self) -> Result<()> {
        let Some(window_app) = &mut self.window_app else {
            return Ok(());
        };
        let context = window_app.as_context(&self.instance);
        window_app.simulation.step(context);
        window_app
            .renderer
            .render(window_app.simulation.points(), context)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("no compatible adapter")]
pub struct NoAdapter;

#[derive(Debug, Clone, Copy, Error)]
#[error("no compatible device")]
pub struct NoDevice;

impl WindowApp {
    async fn new(
        instance: &Instance,
        window: Window,
        points: &[Point],
        maps: &[Map],
    ) -> Result<Self> {
        let window = Arc::new(window);

        let surface = instance.create_surface(Arc::clone(&window))?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .ok_or(NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device"),
                    required_features: Features::PUSH_CONSTANTS,
                    ..Default::default()
                },
                None,
            )
            .await?;

        let surface_capabilities = surface.get_capabilities(&adapter);
        let PhysicalSize { width, height } = window.inner_size();
        let surface_configuration = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_capabilities.formats[0],
            width,
            height,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_configuration);

        let context = Context {
            instance,
            adapter: &adapter,
            device: &device,
            queue: &queue,
            window: &window,
            surface: &surface,
            surface_configuration: &surface_configuration,
        };

        let simulation = Simulation::new(points, maps, context);

        let renderer = Renderer::new(context);

        Ok(WindowApp {
            adapter,
            device,
            queue,
            window,
            surface,
            surface_configuration,
            simulation,
            renderer,
        })
    }

    fn from_event_loop_blocking(
        instance: &Instance,
        event_loop: &ActiveEventLoop,
        size: impl Into<Size>,
        points: &[Point],
        maps: &[Map],
    ) -> Result<Self> {
        let window = event_loop.create_window(WindowAttributes::default().with_inner_size(size))?;
        futures::executor::block_on(WindowApp::new(instance, window, points, maps))
    }

    fn as_context<'app>(&'app self, instance: &'app Instance) -> Context<'app> {
        Context {
            instance,
            adapter: &self.adapter,
            device: &self.device,
            queue: &self.queue,
            window: &self.window,
            surface: &self.surface,
            surface_configuration: &self.surface_configuration,
        }
    }

    fn resize(&mut self) {
        let PhysicalSize { width, height } = self.window.inner_size();
        self.surface_configuration.width = width;
        self.surface_configuration.height = height;
        self.surface
            .configure(&self.device, &self.surface_configuration);
    }
}

impl ApplicationHandler for AppInner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window_app.is_none() {
            match WindowApp::from_event_loop_blocking(
                &self.instance,
                event_loop,
                LogicalSize::new(600, 600),
                &self.points,
                &self.maps,
            ) {
                Err(error) => {
                    error!("failed to create window: {error}");
                    event_loop.exit();
                }
                Ok(window_app) => self.window_app = Some(window_app),
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window_app) = &mut self.window_app else {
            return;
        };

        use WindowEvent as E;
        match event {
            E::Resized(_) => window_app.resize(),
            E::CloseRequested => event_loop.exit(),
            E::RedrawRequested => {
                window_app.window.pre_present_notify();
                if let Err(error) = self.tick() {
                    error!("failed to compute or draw the next animation frame: {error}");
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window_app) = &self.window_app {
            let now = Instant::now();
            if now - self.last_tick >= self.delta_time {
                window_app.window.request_redraw();
                self.last_tick = now;
                event_loop.set_control_flow(ControlFlow::WaitUntil(now + self.delta_time));
            }
        }
    }
}
