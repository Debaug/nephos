use std::iter;

use bytemuck::{Pod, Zeroable};
use glam::{Affine2, Mat3, Vec2, Vec4};
use wgpu::{
    include_wgsl, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderStages,
};

use crate::{app::Context, buffer::Buffer};

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Point {
    pub position: Vec2,
}

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct WgpuMat3x3([Vec4; 3]);

#[derive(Debug)]
pub struct Simulation {
    points: Buffer<Point>,
    _maps: Buffer<WgpuMat3x3>,
    pipeline: ComputePipeline,
    bind_group: BindGroup,
}

impl Simulation {
    pub fn new(
        points: &[Point],
        maps: impl IntoIterator<Item = Affine2>,
        context: Context,
    ) -> Self {
        let points = Buffer::new(
            points,
            Some("Points"),
            BufferUsages::STORAGE | BufferUsages::VERTEX,
            context,
        );

        let maps: Vec<WgpuMat3x3> = maps
            .into_iter()
            .map(|map| {
                let mat: Mat3 = map.into();
                WgpuMat3x3([
                    mat.col(0).extend(0.0),
                    mat.col(1).extend(0.0),
                    mat.col(2).extend(0.0),
                ])
            })
            .collect();
        let maps = Buffer::new(&maps, Some("Maps"), BufferUsages::STORAGE, context);

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Simulation Compute Pipeline Bind Group Layout"),
                    entries: &[
                        // points
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // maps
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Simulation Compute Pipeline Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: points.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: maps.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = context
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Simulation Compute Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader = context
            .device
            .create_shader_module(include_wgsl!("sim.wgsl"));

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Simulation Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            points,
            _maps: maps,
            pipeline,
            bind_group,
        }
    }

    pub fn step(&self, context: Context<'_>) {
        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Simulation Command Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Simulation Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(self.points.len_u32(), 1, 1);
        }

        context.queue.submit(iter::once(encoder.finish()));
    }

    pub fn points(&self) -> &Buffer<Point> {
        &self.points
    }
}
