use std::{borrow::Cow, iter, mem, sync::OnceLock};

use glam::{Affine2, Mat3};
use wgpu::{
    include_wgsl, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState, BufferAddress,
    BufferBindingType, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, FragmentState, LoadOp, Operations, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, ShaderStages, StoreOp, SurfaceTexture, Texture, TextureFormat,
    TextureView, TextureViewDescriptor, VertexBufferLayout, VertexState,
};

use crate::{app::Context, buffer::Buffer, sim::Point, util::WgpuMat3x3};

pub trait RenderTarget: Send + 'static {
    fn texture_view(&self) -> Cow<TextureView>;
}

impl RenderTarget for SurfaceTexture {
    fn texture_view(&self) -> Cow<TextureView> {
        Cow::Owned(self.texture.create_view(&TextureViewDescriptor {
            label: Some("Surface Texture View"),
            ..Default::default()
        }))
    }
}

impl RenderTarget for Texture {
    fn texture_view(&self) -> Cow<TextureView> {
        Cow::Owned(self.create_view(&TextureViewDescriptor {
            label: Some("Texture View"),
            ..Default::default()
        }))
    }
}

impl RenderTarget for TextureView {
    fn texture_view(&self) -> Cow<TextureView> {
        Cow::Borrowed(self)
    }
}

#[derive(Debug, Clone)]
pub struct Renderer {
    pipeline: RenderPipeline,
}

#[derive(Debug)]
pub struct Camera {
    _buffer: Buffer<WgpuMat3x3>,
    bind_group: BindGroup,
}

impl Renderer {
    pub fn new(context: Context, texture_format: TextureFormat) -> Self {
        dbg!(texture_format);

        let pipeline_layout = context
            .device()
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Camera::bind_group_layout(context.borrow())],
                push_constant_ranges: &[],
            });

        let shader = context
            .device()
            .create_shader_module(include_wgsl!("render.wgsl"));

        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: mem::size_of::<Point>() as BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
        };

        let pipeline = context
            .device()
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    buffers: &[vertex_buffer_layout],
                    entry_point: None,
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    targets: &[Some(ColorTargetState {
                        format: texture_format,
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
                    })],
                    entry_point: None,
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::PointList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        Self { pipeline }
    }

    pub fn render<T: RenderTarget>(
        &self,
        points: &Buffer<Point>,
        camera: &Camera,
        target: &T,
        context: Context,
    ) -> wgpu_async::WgpuFuture<()> {
        self.render_all(iter::once((points, camera, target)), context)
    }

    pub fn render_all<'pts, 'cam, 'tgt, T: RenderTarget>(
        &self,
        jobs: impl Iterator<Item = (&'pts Buffer<Point>, &'cam Camera, &'tgt T)>,
        context: Context,
    ) -> wgpu_async::WgpuFuture<()> {
        let commands = jobs.map(|(points, camera, target)| {
            let texture_view = target.texture_view();

            let mut encoder = context
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Render Command Encoder"),
                });
            {
                let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &texture_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });

                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_vertex_buffer(0, *points.slice(..));
                render_pass.set_bind_group(0, &camera.bind_group, &[]);
                render_pass.draw(0..points.len_u32(), 0..1);
            }
            encoder.finish()
        });

        context.queue().submit(commands)
    }
}

impl Camera {
    const BIND_GROUP_LAYOUT_DESCRIPTOR: BindGroupLayoutDescriptor<'static> =
        BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        };

    fn bind_group_layout(context: Context) -> &'static BindGroupLayout {
        static LAYOUT: OnceLock<BindGroupLayout> = OnceLock::new();
        LAYOUT.get_or_init(|| {
            context
                .device()
                .create_bind_group_layout(&Self::BIND_GROUP_LAYOUT_DESCRIPTOR)
        })
    }

    pub fn new(transform: Affine2, context: Context) -> Self {
        let mat = [WgpuMat3x3::from(Mat3::from(transform.inverse()))];
        let bytes = bytemuck::cast_slice(&mat);
        let buffer = Buffer::from_data(
            bytes,
            Some("Camera Buffer"),
            BufferUsages::UNIFORM,
            context.borrow(),
        );

        let bind_group = context.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: Self::bind_group_layout(context.borrow()),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            _buffer: buffer,
            bind_group,
        }
    }
}

impl AsRef<Camera> for Camera {
    fn as_ref(&self) -> &Camera {
        self
    }
}
