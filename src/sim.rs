use std::{future::Future, iter, mem, num::NonZero};

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec2};

use itertools::Itertools;
use wgpu::{
    include_wgsl, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BufferBinding,
    BufferBindingType, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderStages,
};

use crate::{app::Context, buffer::Buffer, map::Map, util::WgpuMat3x3};

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct Point {
    pub position: Vec2,
}

#[derive(Debug)]
pub struct Simulation<P: AsRef<Buffer<Point>>> {
    points: P,
    point_bind_groups: Vec<(BindGroup, u32)>,
    _maps: Buffer<WgpuMat3x3>,
    map_bind_group: BindGroup,
    _map_indices: Buffer<u32>,
    pipeline: ComputePipeline,
}

impl<P: AsRef<Buffer<Point>>> Simulation<P> {
    pub fn new(points: P, maps: &[Map], context: Context) -> Self {
        let points_buf = points.as_ref();

        let (point_bind_group_layout, point_bind_group) =
            Self::point_bind_groups(points_buf, context.borrow());

        let maps_gpu_repr: Vec<WgpuMat3x3> = maps
            .iter()
            .map(|map| {
                let mat: Mat3 = map.map.into();
                WgpuMat3x3::from(mat)
            })
            .collect();
        let map_buffer = Buffer::new(
            &maps_gpu_repr,
            Some("Maps"),
            BufferUsages::STORAGE,
            context.borrow(),
        );

        const MAP_INDEX_ARRAY_LEN: usize = 144;

        let probability_weight_sum: f32 = maps.iter().map(|map| map.probability_weight).sum();
        let probabilities = maps
            .iter()
            .map(|map| map.probability_weight / probability_weight_sum);
        let cumulated_probabilities = probabilities.scan(0.0, |accumulator, probability| {
            *accumulator += probability;
            Some((*accumulator * MAP_INDEX_ARRAY_LEN as f32).round() as usize)
        });
        let map_index_array: Vec<u32> = iter::once(0)
            .chain(cumulated_probabilities)
            .tuple_windows()
            .enumerate()
            .flat_map(|(i, (p, q))| iter::repeat_n(i as u32, q - p))
            .collect();
        let map_indices = Buffer::new(
            &map_index_array,
            Some("Map Indices"),
            BufferUsages::STORAGE,
            context.borrow(),
        );

        let (map_bind_group_layout, map_bind_group) =
            Self::map_bind_group(&map_buffer, &map_indices, context.borrow());

        let pipeline_layout = context
            .device()
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Simulation Compute Pipeline Layout"),
                bind_group_layouts: &[&map_bind_group_layout, &point_bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader = context
            .device()
            .create_shader_module(include_wgsl!("sim.wgsl"));

        let pipeline = context
            .device()
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Simulation Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("step_sim"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            points,
            _maps: map_buffer,
            _map_indices: map_indices,
            pipeline,
            point_bind_groups: point_bind_group,
            map_bind_group,
        }
    }

    fn map_bind_group(
        maps: &Buffer<WgpuMat3x3>,
        map_indices: &Buffer<u32>,
        context: Context,
    ) -> (BindGroupLayout, BindGroup) {
        let map_bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Simulation Compute Pipeline Bind Group Layout for Linear Maps"),
                    entries: &[
                        // maps
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // map indices
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

        let map_bind_group = context.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Simulation Compute Pipeline Bind Group for Linear Maps"),
            layout: &map_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: maps.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: map_indices.as_entire_binding(),
                },
            ],
        });

        (map_bind_group_layout, map_bind_group)
    }

    fn point_bind_groups(
        points: &Buffer<Point>,
        context: Context,
    ) -> (BindGroupLayout, Vec<(BindGroup, u32)>) {
        const MAX_WORKGROUPS_PER_DISPATCH_UNALIGNED: u32 = u16::MAX as u32;
        let alignment = context
            .device()
            .limits()
            .min_storage_buffer_offset_alignment;
        let max_workgroups_per_dispatch =
            (MAX_WORKGROUPS_PER_DISPATCH_UNALIGNED / alignment) * alignment;

        let n_max = points.len_u32() / max_workgroups_per_dispatch;
        let rem = points.len_u32() % max_workgroups_per_dispatch;

        let point_bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Simulation Compute Pipeline Bind Group Layout for Points"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let point_bind_groups = iter::repeat_n(max_workgroups_per_dispatch, n_max as usize)
            .chain([rem])
            .scan(0, |start, len| {
                let this_start = *start;
                *start += len;
                Some((this_start, len))
            })
            .enumerate()
            .map(|(idx, (start, len))| {
                let bind_group = context.device().create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!(
                        "Simulation Compute Pipeline Bind Group for Points (Chunk #{idx})"
                    )),
                    layout: &point_bind_group_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: points,
                            offset: u64::from(start) * mem::size_of::<Point>() as u64,
                            size: NonZero::new(u64::from(len) * mem::size_of::<Point>() as u64),
                        }),
                    }],
                });
                (bind_group, len)
            })
            .collect();

        (point_bind_group_layout, point_bind_groups)
    }

    pub fn step(&self, context: Context<'_>) -> impl Future<Output = ()> + 'static {
        let commands =
            self.point_bind_groups
                .iter()
                .enumerate()
                .map(|(idx, &(ref point_bind_group, len))| {
                    let mut encoder =
                        context
                            .device()
                            .create_command_encoder(&CommandEncoderDescriptor {
                                label: Some(&format!(
                                    "Simulation Command Encoder for Chunk #{idx}"
                                )),
                            });

                    {
                        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: Some(&format!("Simulation Compute Pass for Chunk #{idx}")),
                            timestamp_writes: None,
                        });

                        compute_pass.set_pipeline(&self.pipeline);
                        compute_pass.set_bind_group(0, &self.map_bind_group, &[]);
                        compute_pass.set_bind_group(1, point_bind_group, &[]);
                        compute_pass.dispatch_workgroups(len, 1, 1);
                    }

                    encoder.finish()
                });

        context.queue().submit(commands)
    }

    pub fn points(&self) -> &P {
        &self.points
    }
}
