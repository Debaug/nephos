use std::{iter, mem};

use color_eyre::eyre::Result;
use wgpu::{
    include_wgsl, BlendState, BufferAddress, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, FragmentState, LoadOp, Operations, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, StoreOp, TextureViewDescriptor, VertexBufferLayout, VertexState,
};

use crate::{app::Context, buffer::Buffer, sim::Point};

#[derive(Debug, Clone)]
pub struct Renderer {
    pipeline: RenderPipeline,
}

impl Renderer {
    pub fn new(context: Context) -> Self {
        let pipeline_layout = context
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let shader = context
            .device
            .create_shader_module(include_wgsl!("render.wgsl"));

        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: mem::size_of::<Point>() as BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
        };

        let pipeline = context
            .device
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
                        format: context.surface_configuration.format,
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

    pub fn render(&self, points: &Buffer<Point>, context: Context) -> Result<()> {
        let surface_texture = context.surface.get_current_texture()?;
        let surface_texture_view = surface_texture.texture.create_view(&TextureViewDescriptor {
            label: Some("Surface Output Texture View"),
            ..Default::default()
        });

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Command Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, points.slice(..));
            render_pass.draw(0..points.len_u32(), 0..1);
        }

        context.queue.submit(iter::once(encoder.finish()));
        surface_texture.present();

        Ok(())
    }
}
