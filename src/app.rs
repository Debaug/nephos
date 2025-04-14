use std::{
    borrow::Cow,
    mem,
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    time::Duration,
};

use color_eyre::eyre::Result;
use futures::future::BoxFuture;
use log::error;
use thiserror::Error;
use tokio::runtime::Runtime;
use wgpu::{
    Adapter, Backends, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    RequestAdapterOptions, Surface, SurfaceConfiguration, SurfaceTexture, TextureUsages,
};
use wgpu_async::{AsyncDevice, AsyncQueue};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

pub trait AppBuilder: Send + 'static {
    type App: App;
    fn build(
        self,
        surface_configuration: &SurfaceConfiguration,
        context: Context,
    ) -> BoxFuture<'static, Result<Self::App>>;
}

pub trait App: Send + 'static {
    fn event(&mut self, event: WindowEvent, context: Context, controller: LocalAppController);

    fn render(&mut self, target: &SurfaceTexture, context: Context) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct AppController {
    exit_tx: mpsc::Sender<()>,
}

#[derive(Debug, Clone)]
pub struct LocalAppController<'a> {
    exit_tx: mpsc::Sender<()>,
    event_loop: &'a ActiveEventLoop,
}

impl LocalAppController<'_> {
    pub fn exit(&self) {
        self.event_loop.exit();
    }

    pub fn into_non_local(self) -> AppController {
        AppController {
            exit_tx: self.exit_tx,
        }
    }

    pub fn to_non_local(&self) -> AppController {
        self.clone().into_non_local()
    }
}

impl AppController {
    pub fn exit(&self) {
        self.exit_tx.send(()).expect("failed to send exit message");
    }
}

#[derive(Debug, Clone)]
struct ContextInner {
    runtime: Arc<Runtime>,
    instance: Instance,
    adapter: Adapter,
    device: AsyncDevice,
    queue: AsyncQueue,
}

#[derive(Debug)]
pub struct Context<'a> {
    inner: Cow<'a, ContextInner>,
}

impl<'a> Context<'a> {
    fn borrowed(inner: &'a ContextInner) -> Self {
        Self {
            inner: Cow::Borrowed(inner),
        }
    }

    pub fn borrow<'b>(&'b self) -> Context<'b>
    where
        'a: 'b,
    {
        Context {
            inner: Cow::Borrowed(&self.inner),
        }
    }

    pub fn to_static(&self) -> Context<'static> {
        self.borrow().into_static()
    }

    pub fn into_static(self) -> Context<'static> {
        Context {
            inner: Cow::Owned(self.inner.into_owned()),
        }
    }

    pub fn runtime(&self) -> &Runtime {
        &self.inner.runtime
    }

    pub fn instance(&self) -> &Instance {
        &self.inner.instance
    }

    pub fn adapter(&self) -> &Adapter {
        &self.inner.adapter
    }

    pub fn device(&self) -> &AsyncDevice {
        &self.inner.device
    }

    pub fn queue(&self) -> &AsyncQueue {
        &self.inner.queue
    }
}

#[derive(Debug, Clone)]
pub struct Run<A: AppBuilder> {
    pub app_builder: A,
    pub window_attributes: WindowAttributes,
    pub features: Features,
    pub limits: Limits,
    pub surface_usages: TextureUsages,
}

impl<A: AppBuilder> Run<A> {
    pub fn new(app_builder: A) -> Self {
        Self {
            app_builder,
            window_attributes: Default::default(),
            features: Default::default(),
            limits: Default::default(),
            surface_usages: TextureUsages::RENDER_ATTACHMENT,
        }
    }

    pub fn with_window_attributes(mut self, window_attributes: WindowAttributes) -> Self {
        self.window_attributes = window_attributes;
        self
    }

    pub fn with_features(mut self, features: Features) -> Self {
        self.features = features;
        self
    }

    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_surface_usages(mut self, usages: TextureUsages) -> Self {
        self.surface_usages = usages;
        self
    }

    pub fn run(self) -> Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        Ok(event_loop.run_app(&mut AppContainer::Created(CreatedApp {
            run: self,
            runtime: Arc::new(Runtime::new()?),
        }))?)
    }
}

#[derive(Debug)]
enum AppContainer<A: AppBuilder> {
    Error,
    Created(CreatedApp<A>),
    // Spawned(Receiver<Result<ReadyAppContainer<A::App>>>),
    Ready(ReadyAppContainer<A::App>),
}

#[derive(Debug)]
struct CreatedApp<A: AppBuilder> {
    run: Run<A>,
    runtime: Arc<Runtime>,
}

#[derive(Debug)]
struct WindowSurface {
    window: Arc<Window>,
    surface: Surface<'static>,
    surface_configuration: SurfaceConfiguration,
}

#[derive(Debug)]
struct ReadyAppContainer<A: App> {
    context: ContextInner,
    window: WindowSurface,
    app: A,
    exit_tx: Sender<()>,
    exit_rx: Receiver<()>,
}

impl<A: AppBuilder> ApplicationHandler for AppContainer<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self, AppContainer::Created(_)) {
            return;
        }

        let Self::Created(CreatedApp { run, runtime }) = mem::replace(self, Self::Error) else {
            unreachable!()
        };

        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let window = match event_loop.create_window(run.window_attributes.clone()) {
            Err(error) => {
                error!("failed to create window: {error}");
                event_loop.exit();
                return;
            }
            Ok(window) => Arc::new(window),
        };

        let surface = match instance.create_surface(window.clone()) {
            Err(error) => {
                error!("failed to create surface: {error}");
                event_loop.exit();
                return;
            }
            Ok(surface) => surface,
        };

        *self = AppContainer::Ready(
            futures::executor::block_on(ReadyAppContainer::new(
                run, instance, surface, window, runtime,
            ))
            .expect("failed to create app"),
        );
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Self::Ready(ReadyAppContainer {
            context,
            window,
            app,
            exit_tx,
            exit_rx,
        }) = self
        else {
            return;
        };

        if exit_rx.try_recv().is_ok() {
            event_loop.exit();
            return;
        }

        let context = Context::borrowed(context);

        match event {
            WindowEvent::Resized(new_size) => {
                window.surface_configuration.width = new_size.width;
                window.surface_configuration.height = new_size.height;
                window
                    .surface
                    .configure(context.device(), &window.surface_configuration);
            }

            WindowEvent::RedrawRequested => {
                let surface = match window.surface.get_current_texture() {
                    Err(error) => {
                        error!("failed to acquire Surface current texture: {error}");
                        event_loop.exit();
                        return;
                    }
                    Ok(surface) => surface,
                };
                window.window.pre_present_notify();
                match app.render(&surface, context.borrow()) {
                    Err(error) => {
                        error!("failed to render app: {error}");
                        event_loop.exit();
                        return;
                    }
                    Ok(_) => {
                        surface.present();
                    }
                }

                let window = window.window.clone();
                context.runtime().spawn(async move {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    window.request_redraw();
                });
            }

            _ => {}
        }

        let exit_tx = exit_tx.clone();
        let controller = LocalAppController {
            exit_tx,
            event_loop,
        };
        app.event(event, context, controller);
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("no compatible adapter")]
pub struct NoAdapter;

#[derive(Debug, Clone, Copy, Error)]
#[error("no compatible device")]
pub struct NoDevice;

impl<A: App> ReadyAppContainer<A> {
    pub async fn new<B: AppBuilder<App = A>>(
        run: Run<B>,
        instance: Instance,
        surface: Surface<'static>,
        window: Arc<Window>,
        runtime: Arc<Runtime>,
    ) -> Result<Self> {
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
                    required_features: run.features,
                    required_limits: run.limits,
                    ..Default::default()
                },
                None,
            )
            .await?;
        let (device, queue) = wgpu_async::wrap(Arc::new(device), Arc::new(queue));

        let surface_capabilities = surface.get_capabilities(&adapter);
        let PhysicalSize { width, height } = window.inner_size();
        let surface_configuration = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT | run.surface_usages,
            format: surface_capabilities.formats[0],
            width,
            height,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_configuration);

        let context = ContextInner {
            runtime,
            instance,
            adapter,
            device,
            queue,
        };

        let app = run
            .app_builder
            .build(&surface_configuration, Context::borrowed(&context))
            .await?;

        let (exit_tx, exit_rx) = mpsc::channel();

        Ok(Self {
            context,
            window: WindowSurface {
                window,
                surface,
                surface_configuration,
            },
            app,
            exit_tx,
            exit_rx,
        })
    }
}
