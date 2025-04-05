use std::sync::Arc;

use color_eyre::eyre::Result;
use log::error;
use thiserror::Error;
use tokio::task::{self, JoinHandle};
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor, Queue,
    RequestAdapterOptions, RequestAdapterOptionsBase, Surface, SurfaceConfiguration, TextureUsages,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{self, ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
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
}

#[derive(Debug)]
struct WindowApp {
    adapter: Adapter,
    device: Device,
    queue: Queue,
    window: Arc<Window>,
    surface_configuration: SurfaceConfiguration,
    surface: Surface<'static>,
}

#[derive(Debug)]
pub struct Context<'app> {
    instance: &'app Instance,
    adapter: &'app Adapter,
    device: &'app Device,
    queue: &'app Queue,
    window: &'app Window,
    surface: &'app Surface<'static>,
    surface_configuration: &'app SurfaceConfiguration,
}

impl App {
    pub fn new() -> Result<Self> {
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
            },
        })
    }

    pub fn run(mut self) -> Result<()> {
        self.event_loop.run_app(&mut self.inner)?;
        Ok(())
    }

    fn context(&self) -> Option<Context> {
        let window_app = self.inner.window_app.as_ref()?;
        Some(Context {
            instance: &self.inner.instance,
            adapter: &window_app.adapter,
            device: &window_app.device,
            queue: &window_app.queue,
            window: &window_app.window,
            surface: &window_app.surface,
            surface_configuration: &window_app.surface_configuration,
        })
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("no adapter")]
pub struct NoAdapter;

impl WindowApp {
    async fn new(instance: &Instance, window: Window) -> Result<Self> {
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
                    label: Some("device"),
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

        Ok(WindowApp {
            adapter,
            device,
            queue,
            window,
            surface,
            surface_configuration,
        })
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
        if self.window_app.is_some() {
            return;
        }

        let window = match event_loop
            .create_window(WindowAttributes::default().with_inner_size(PhysicalSize::new(800, 800)))
        {
            Ok(window) => window,
            Err(err) => {
                error!("failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        match futures::executor::block_on(WindowApp::new(&self.instance, window)) {
            Ok(window_app) => self.window_app = Some(window_app),
            Err(err) => {
                error!("failed to initialize app: {err}");
                event_loop.exit();
            }
        };
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
            _ => {}
        }
    }
}
