use std::{
    cmp, f32,
    fmt::Debug,
    future::{self, Future},
    iter, mem,
    num::NonZero,
    ops::Deref,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use bytemuck::{Pod, Zeroable};
use color_eyre::eyre::{Ok, Result};
use futures::FutureExt;
use glam::{vec2, Affine2, Mat2, Mat3, Vec2};
use image::{GrayImage, RgbaImage};
use itertools::Itertools;
use log::info;
use rand::Rng;
use wgpu::*;
use wgsl_preprocessor::ShaderBuilder;

use crate::{
    app::Context,
    buffer::Buffer,
    map::Map,
    render::{Camera, Renderer},
    sim::Point,
    util::{mat2, SyncingFuture, WgpuMat3x3},
};

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct AffineDecomposition {
    angle: f32,
    shear: f32,
    scale: Vec2,
    translation: Vec2,
}

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct Affine {
    decomposition: AffineDecomposition,
    _padding: [u8; 8],
    computed: WgpuMat3x3,
}

impl AffineDecomposition {
    fn compose(self) -> Affine {
        let rotation = Mat2::from_angle(self.angle);

        // let shear = mat2(
        //     1.0, self.shear, //
        //     0.0, 1.0,
        // );

        let scale = Mat2::from_diagonal(self.scale.max(vec2(0.2, 0.2)));

        let affine = Affine2::from_mat2_translation(rotation * scale, self.translation);

        Affine {
            decomposition: self,
            _padding: [0; 8],
            computed: WgpuMat3x3::from(Mat3::from(affine)),
        }
    }
}

const IMAGE_SIZE: u32 = 256;
const IMAGE_FORMAT: TextureFormat = TextureFormat::R8Unorm;
const IMAGE_BYTES_PER_ROW: u32 = 256;
const RENDER_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
const RENDER_TEXTURE_BYTES_PER_ROW: u32 = 256 * 4;

#[derive(Debug)]
pub struct Evolver {
    step: AtomicUsize,
    maps_per_set: usize,
    elite_len: usize,
    depth: usize,
    n_children: usize,
    mutation_strength: f32,
    mutation_damping: f32,
    maps: Arc<Buffer<Affine>>,
    random_points: Buffer<Point>,
    simulate: Simulate,
    renderer: Renderer,
    camera: Camera,
    rate: Rate,
    select: Select,
}

impl Evolver {
    pub fn new(
        source: &GrayImage,
        maps_per_set: usize,
        elite_len: usize,
        depth: usize,
        n_children: usize,
        n_points: usize,
        mutation_strength: f32,
        mutation_damping: f32,
        context: Context,
    ) -> Result<Self> {
        assert_eq!(source.width(), IMAGE_SIZE);
        assert_eq!(source.height(), IMAGE_SIZE);
        assert!(source.pixels().all(|p| [0, 255].contains(&p.0[0])));

        let texture_size = Extent3d {
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
            depth_or_array_layers: 1,
        };
        let source_texture = context.device().create_texture(&TextureDescriptor {
            label: Some("Evolution Source Image Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: IMAGE_FORMAT,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[IMAGE_FORMAT],
        });
        context.queue().write_texture(
            TexelCopyTextureInfo {
                texture: &source_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(source.as_raw().as_slice()),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(IMAGE_BYTES_PER_ROW),
                rows_per_image: None,
            },
            texture_size,
        );

        let source_view = source_texture.create_view(&TextureViewDescriptor {
            label: Some("Evolution Source Image Texture View"),
            ..Default::default()
        });

        let width = elite_len * (1 + n_children);

        let mut rng = rand::rng();
        let mut random_parameter = || rng.random_range(-1.0..=1.0);

        let maps = iter::repeat_with(|| {
            AffineDecomposition {
                angle: random_parameter() * f32::consts::PI,
                shear: random_parameter(),
                // shear: 0.0,
                scale: vec2(
                    random_parameter() * 0.5 + 0.5,
                    random_parameter() * 0.5 + 0.5,
                ),
                translation: vec2(random_parameter(), random_parameter()),
            }
            .compose()
        })
        .take(width * maps_per_set)
        .collect_vec();
        let maps = Buffer::from_data(
            &maps,
            Some("Evolving Maps"),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            context.borrow(),
        );

        let random_points = iter::from_fn(|| {
            Some(Point {
                position: vec2(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)),
            })
        })
        .take(n_points)
        .collect_vec();
        let random_points = Buffer::from_data(
            &random_points,
            Some("Evolution Random Point Buffer"),
            BufferUsages::COPY_SRC,
            context.borrow(),
        );

        let simulate = Simulate::new(&maps, width, n_points, context.borrow())?;

        let renderer = Renderer::new(context.borrow(), RENDER_TEXTURE_FORMAT);
        let camera = Camera::new(Affine2::IDENTITY, context.borrow());

        let rate = Rate::new(
            &source_view,
            simulate.point_buffers.iter().map(|(.., view)| view),
            context.borrow(),
        )?;

        let sort = Select::new(width, context.borrow())?;

        Ok(Self {
            step: AtomicUsize::new(0),
            maps_per_set,
            elite_len,
            depth,
            n_children,
            mutation_strength,
            mutation_damping,
            maps: Arc::new(maps),
            random_points,
            simulate,
            renderer,
            camera,
            rate,
            select: sort,
        })
    }

    pub fn step(&self, context: Context) -> impl Future<Output = ()> + 'static {
        self.reset_simulations(context.borrow()).ignore();
        for _ in 0..self.depth {
            self.step_simulations(context.borrow()).ignore();
        }
        self.render_simulations(context.borrow()).ignore();
        self.rate_simulations(context.borrow()).ignore();
        self.select_simulations(context)
    }

    pub async fn evolve(&self, generations: usize, context: Context<'static>) {
        for i in 0..generations {
            info!("Generation {}...", i + 1);
            self.step(context.borrow()).await;
        }
    }

    pub fn reset_simulations(&self, context: Context) -> impl SyncingFuture {
        self.simulate.reset(&self.random_points, context)
    }

    pub fn step_simulations(&self, context: Context) -> impl SyncingFuture {
        self.simulate.step_simulations(
            self.maps_per_set,
            self.step.load(Ordering::Relaxed),
            context,
        )
    }

    pub fn render_simulations(&self, context: Context) -> impl SyncingFuture {
        self.simulate.render(&self.renderer, &self.camera, context)
    }

    pub fn rate_simulations(&self, context: Context) -> impl SyncingFuture {
        self.compare(context.borrow()).ignore();
        self.reduce(context.borrow())
    }

    pub fn compare(&self, context: Context) -> impl SyncingFuture {
        self.rate.compare_all(context.borrow())
    }

    pub fn reduce(&self, context: Context) -> impl SyncingFuture {
        self.rate.reduce_all(context.borrow())
    }

    pub fn select_simulations(&self, context: Context) -> impl Future<Output = ()> + 'static {
        let intermediate_buffers = self
            .rate
            .map_set_data
            .iter()
            .map(|data| {
                data.reduce_buffer_pair
                    .lock()
                    .expect("failed to lock mutex")
                    .a
                    .clone()
            })
            .collect_vec();

        let mutation_strength = self.mutation_strength
            * f32::exp(-(self.step.load(Ordering::Relaxed) as f32 * self.mutation_damping));

        self.select.select(
            self.elite_len,
            self.maps_per_set,
            self.n_children,
            mutation_strength,
            self.maps.clone(),
            intermediate_buffers,
            context,
        )
    }

    pub fn debug_maps(&self, context: Context) -> impl Future<Output = ()> + 'static {
        let data_fut = self.maps.download(context);
        async move {
            dbg!(data_fut.await);
        }
    }

    pub fn download_first_texture(&self, context: Context) -> impl Future<Output = RgbaImage> {
        self.simulate.download_first_texture(context)
    }

    pub fn debug_first_points(&self, context: Context) -> impl Future<Output = ()> + 'static {
        let data_fut = self.simulate.point_buffers[0].0.download(context);
        async move {
            dbg!(data_fut.await);
        }
    }

    pub fn get_first_buffer_a(&self, context: Context) -> impl Future<Output = Vec<u32>> {
        self.rate.map_set_data[0]
            .reduce_buffer_pair
            .lock()
            .expect("failed to lock mutex")
            .a
            .download(context)
            .then(|buf| {
                future::ready({
                    dbg!(buf.len());
                    buf
                })
            })
    }

    pub fn debug_scores(&self, context: Context) -> impl Future<Output = ()> + 'static {
        let data_fut = self.select.scores.download(context);
        async move {
            dbg!(data_fut.await);
        }
    }

    pub fn get_some_points(&self) -> &Buffer<Point> {
        &self.simulate.point_buffers[0].0
    }

    pub fn get_best_map_set(&self, context: Context) -> impl Future<Output = Vec<Map>> + 'static {
        let fut = self.maps.download(context);
        let maps_per_set = self.maps_per_set;
        async move {
            fut.await[0..maps_per_set]
                .iter()
                .map(|affine| {
                    let mat3 = affine.computed.into();
                    let affine2 = Affine2::from_mat3(mat3);
                    Map {
                        map: affine2,
                        probability_weight: 1.0,
                    }
                })
                .collect_vec()
        }
    }
}

#[derive(Debug, Clone)]
struct Prime {
    bind_group: BindGroup,
    pipeline: ComputePipeline,
}

impl Prime {
    fn new(maps: &Buffer<Affine>, context: Context) -> Result<Self> {
        let map_bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Evolving Map Bind Group"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZero::new(maps.size()),
                        },
                        count: None,
                    }],
                });

        let map_bind_group = context.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Evolving Map Bind Group"),
            layout: &map_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: maps.as_entire_binding(),
            }],
        });

        let shader_builder = ShaderBuilder::new("src/image/prime.wgsl")?;
        let shader = context
            .device()
            .create_shader_module(shader_builder.build());

        let pipeline_layout = context
            .device()
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Evolution Primer Pipeline Layout"),
                bind_group_layouts: &[&map_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = context
            .device()
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Evolution Primer Pipeline Layout"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: None,
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(Self {
            bind_group: map_bind_group,
            pipeline,
        })
    }

    fn run(&self, maps: &Buffer<Affine>, context: Context) -> impl SyncingFuture {
        let mut encoder = context
            .device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Evolution Prime Command Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Evolution Prime Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(maps.len_u32(), 1, 1);
        }
        context.queue().submit(iter::once(encoder.finish()))
    }
}

#[derive(Debug)]
struct Simulate {
    map_set_bind_group: BindGroup,
    point_buffers: Vec<(Buffer<Point>, BindGroup, Texture, TextureView)>,
    simulate_pipeline: ComputePipeline,
}

impl Simulate {
    fn new(
        maps_buffer: &Buffer<Affine>,
        width: usize,
        n_points: usize,
        context: Context,
    ) -> Result<Self> {
        let map_set_bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Evolution Rating Map Set Bind Group Layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let map_set_bind_group = context.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Evolution Rating Map Set Bind Group"),
            layout: &map_set_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: maps_buffer.as_entire_binding(),
            }],
        });

        let point_buffer_bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Evolution Rating Point Buffer Bind Group Layout"),
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

        let point_buffers = (0..width)
            .map(|idx| {
                let buffer = Buffer::new(
                    n_points,
                    Some(&format!("Evolution Rating Point Buffer #{idx}")),
                    BufferUsages::STORAGE
                        | BufferUsages::VERTEX
                        | BufferUsages::COPY_DST
                        | BufferUsages::COPY_SRC,
                    context.borrow(),
                );

                let bind_group = context.device().create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("Evolution Rating Point Buffer #{idx} Bind Group")),
                    layout: &point_buffer_bind_group_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });

                let texture = context.device().create_texture(&TextureDescriptor {
                    label: Some(&format!("Evolution Rating Render Texture #{idx}")),
                    size: Extent3d {
                        width: IMAGE_SIZE,
                        height: IMAGE_SIZE,
                        depth_or_array_layers: 1,
                    },
                    dimension: TextureDimension::D2,
                    format: RENDER_TEXTURE_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::COPY_SRC
                        | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[RENDER_TEXTURE_FORMAT],
                    mip_level_count: 1,
                    sample_count: 1,
                });

                let texture_view = texture.create_view(&TextureViewDescriptor {
                    label: Some(&format!("Evolution Rating Render Texture #{idx} View")),
                    ..Default::default()
                });

                (buffer, bind_group, texture, texture_view)
            })
            .collect();

        let simulate_shader_builder = ShaderBuilder::new("src/image/simulate.wgsl")?;
        let simulate_shader = context
            .device()
            .create_shader_module(simulate_shader_builder.build());

        let simulate_pipeline_layout =
            context
                .device()
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Evolution Simulation Pipeline Layout"),
                    bind_group_layouts: &[
                        &map_set_bind_group_layout,
                        &point_buffer_bind_group_layout,
                    ],
                    push_constant_ranges: &[PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..8,
                    }],
                });

        let simulate_pipeline =
            context
                .device()
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Evolution Simulation Pipeline"),
                    layout: Some(&simulate_pipeline_layout),
                    module: &simulate_shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                });

        Ok(Self {
            map_set_bind_group,
            point_buffers,
            simulate_pipeline,
        })
    }

    fn reset(&self, points: &Buffer<Point>, context: Context) -> impl SyncingFuture {
        let commands = self.point_buffers.iter().map(|(buffer, ..)| {
            let mut encoder = context
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Evolution Point Buffer Reset Command Encoder"),
                });
            encoder.copy_buffer_to_buffer(points, 0, buffer, 0, points.size());
            encoder.finish()
        });
        context.queue().submit(commands)
    }

    fn step_simulations(
        &self,
        maps_per_set: usize,
        step: usize,
        context: Context,
    ) -> impl SyncingFuture {
        let commands = self.point_buffers.iter().enumerate().map(
            |(idx, (point_buffer, point_bind_group, ..))| {
                let mut encoder =
                    context
                        .device()
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some(&format!("Evolution Simulation #{idx} (step #{step})",)),
                        });
                {
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some(&format!(
                            "Evolution Simulation #{idx} (step #{step}) Compute Pass",
                        )),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&self.simulate_pipeline);
                    compute_pass
                        .set_push_constants(0, bytemuck::bytes_of(&((idx * maps_per_set) as u32)));
                    compute_pass.set_push_constants(4, bytemuck::bytes_of(&(maps_per_set as u32)));
                    compute_pass.set_bind_group(0, &self.map_set_bind_group, &[]);
                    compute_pass.set_bind_group(1, point_bind_group, &[]);
                    compute_pass.dispatch_workgroups(point_buffer.len_u32(), 1, 1);
                }
                encoder.finish()
            },
        );
        context.queue().submit(commands)
    }

    fn render(&self, renderer: &Renderer, camera: &Camera, context: Context) -> impl SyncingFuture {
        renderer.render_all(
            self.point_buffers
                .iter()
                .map(|(buffer, .., texture_view)| (buffer, camera, texture_view)),
            context,
        )
    }

    fn download_first_texture(&self, context: Context) -> impl Future<Output = RgbaImage> {
        let copy_buffer = context.device().create_buffer(&BufferDescriptor {
            label: None,
            size: IMAGE_SIZE as u64 * IMAGE_SIZE as u64 * 4,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut copy_encoder = context
            .device()
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        copy_encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &self.point_buffers[0].2,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &copy_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(RENDER_TEXTURE_BYTES_PER_ROW),
                    rows_per_image: None,
                },
            },
            Extent3d {
                width: IMAGE_SIZE,
                height: IMAGE_SIZE,
                depth_or_array_layers: 1,
            },
        );
        let command = copy_encoder.finish();

        let context = context.into_static();
        async move {
            context.queue().submit(iter::once(command)).await;
            let slice = copy_buffer.slice(..);
            slice
                .map_async(wgpu::MapMode::Read)
                .await
                .expect("failed to map buffer");
            let image = {
                let mapped_range = slice.get_mapped_range();
                RgbaImage::from_vec(IMAGE_SIZE, IMAGE_SIZE, mapped_range.to_vec())
                    .expect("failed to create image")
            };
            copy_buffer.unmap();
            image
        }
    }
}

#[derive(Debug)]
struct Rate {
    compare_pipeline: ComputePipeline,
    compare_bind_group_0: BindGroup,
    reduce_pipeline: ComputePipeline,
    map_set_data: Vec<RateMapSet>,
}

#[derive(Debug)]
struct RateMapSet {
    compare_bind_group_1: BindGroup,
    reduce_buffer_pair: Mutex<ReduceBufferPair>,
}

#[derive(Debug)]
struct ReduceBufferPair {
    a: Buffer<u32>,
    b: Buffer<u32>,
    a_to_b: BindGroup,
    b_to_a: BindGroup,
}

impl ReduceBufferPair {
    fn swap(&mut self) {
        mem::swap(&mut self.a, &mut self.b);
        mem::swap(&mut self.a_to_b, &mut self.b_to_a);
    }
}

impl Rate {
    fn new<'a>(
        source: &TextureView,
        render_textures: impl IntoIterator<Item = &'a TextureView>,
        context: Context,
    ) -> Result<Self> {
        let sampler = context.device().create_sampler(&SamplerDescriptor {
            label: Some("Evolution Comparison Sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let compare_bind_group_0_layout =
            context
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Evolution Comparison Bind Group 0 Layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: false },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        let compare_bind_group_0 = context.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Evolution Comparison Bind Group 0"),
            layout: &compare_bind_group_0_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(source),
                },
            ],
        });

        let compare_bind_group_1_layout =
            context
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Evolution Comparison Bind Group 1 Layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: false },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let compare_shader_builder = ShaderBuilder::new("src/image/score/compare.wgsl")?;
        let compare_shader = context
            .device()
            .create_shader_module(compare_shader_builder.build());

        let compare_pipeline_layout =
            context
                .device()
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Evolution Comparison Pipeline Layout"),
                    bind_group_layouts: &[
                        &compare_bind_group_0_layout,
                        &compare_bind_group_1_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let compare_pipeline =
            context
                .device()
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Evolution Comparison Pipeline"),
                    layout: Some(&compare_pipeline_layout),
                    module: &compare_shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });

        let reduce_bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Evolution Reducting Bind Group Layout"),
                    entries: &[
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
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let reduce_shader_builder = ShaderBuilder::new("src/image/score/reduce.wgsl")?;
        let reduce_shader = context
            .device()
            .create_shader_module(reduce_shader_builder.build());

        let reduce_pipeline_layout =
            context
                .device()
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Evolution Reduction Pipeline Layout"),
                    bind_group_layouts: &[&reduce_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let reduce_pipeline =
            context
                .device()
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Evolution Reduction Pipeline"),
                    layout: Some(&reduce_pipeline_layout),
                    module: &reduce_shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });

        let map_set_data = render_textures
            .into_iter()
            .enumerate()
            .map(|(idx, render_texture_view)| {
                let buf_len = IMAGE_SIZE * IMAGE_SIZE;
                let buffer_a = Buffer::new(
                    buf_len as usize,
                    Some(&format!("Evolution Reducing Cache #{idx}A")),
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    context.borrow(),
                );
                let buffer_b = Buffer::new(
                    buf_len as usize,
                    Some(&format!("Evolution Reduction Cache #{idx}B")),
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    context.borrow(),
                );

                let compare_bind_group_1 =
                    context.device().create_bind_group(&BindGroupDescriptor {
                        label: Some(&format!("Evolution Comparison Bind Group #{idx}")),
                        layout: &compare_bind_group_1_layout,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::TextureView(render_texture_view),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: buffer_a.as_entire_binding(),
                            },
                        ],
                    });

                let a_to_b = context.device().create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("Evolution Reduction Bind Group #{idx} A->B")),
                    layout: &reduce_bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: buffer_a.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer_b.as_entire_binding(),
                        },
                    ],
                });

                let b_to_a = context.device().create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("Evolution Reduction Bind Group #{idx} B->A")),
                    layout: &reduce_bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: buffer_b.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer_a.as_entire_binding(),
                        },
                    ],
                });

                RateMapSet {
                    compare_bind_group_1,
                    reduce_buffer_pair: Mutex::new(ReduceBufferPair {
                        a: buffer_a,
                        b: buffer_b,
                        a_to_b,
                        b_to_a,
                    }),
                }
            })
            .collect_vec();

        Ok(Self {
            compare_pipeline,
            compare_bind_group_0,
            reduce_pipeline,
            map_set_data,
        })
    }

    fn compare_all(&self, context: Context) -> impl SyncingFuture {
        let commands = self
            .map_set_data
            .iter()
            .enumerate()
            .map(|(idx, map_set_data)| {
                let mut encoder =
                    context
                        .device()
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some(&format!("Evolution Comparison Command Encoder #{idx}")),
                        });
                {
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some(&format!("Evolution Comparison Compute Pass #{idx}")),
                        timestamp_writes: None,
                    });
                    compute_pass.set_pipeline(&self.compare_pipeline);
                    compute_pass.set_bind_group(0, &self.compare_bind_group_0, &[]);
                    compute_pass.set_bind_group(1, &map_set_data.compare_bind_group_1, &[]);
                    compute_pass.dispatch_workgroups(IMAGE_SIZE / 8, IMAGE_SIZE / 8, 1);
                }
                encoder.finish()
            });
        context.queue().submit(commands)
    }

    fn reduce_all_once(&self, context: Context, n: u32) -> impl SyncingFuture {
        let commands = self
            .map_set_data
            .iter()
            .enumerate()
            .map(|(idx, map_set_data)| {
                let mut encoder =
                    context
                        .device()
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some(&format!("Evolution Reduction Command Encoder #{idx}")),
                        });
                {
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some(&format!("Evolution Reduction Compute Pass #{idx}")),
                        timestamp_writes: None,
                    });
                    let mut reduce_buffer_pair = map_set_data
                        .reduce_buffer_pair
                        .lock()
                        .expect("failed to lock mutex");
                    compute_pass.set_pipeline(&self.reduce_pipeline);
                    compute_pass.set_bind_group(0, &reduce_buffer_pair.a_to_b, &[]);
                    compute_pass.dispatch_workgroups((n + 63) / 64, 1, 1);
                    reduce_buffer_pair.swap();
                }
                encoder.finish()
            });
        context.queue().submit(commands)
    }

    fn reduce_all(&self, context: Context) -> impl SyncingFuture {
        let ns =
            itertools::iterate(IMAGE_SIZE * IMAGE_SIZE / 2, |&n| n / 2).take_while(|&n| n != 0);

        ns.map(|n| self.reduce_all_once(context.borrow(), n))
            .last()
            .unwrap()
    }
}

#[derive(Debug)]
struct Select {
    scores: Buffer<u32>,
}

impl Select {
    fn new(width: usize, context: Context) -> Result<Self> {
        let scores = Buffer::new(
            width,
            Some("Evolution Score Buffer"),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            context.borrow(),
        );

        Ok(Self { scores })
    }

    fn select(
        &self,
        elite_len: usize,
        maps_per_set: usize,
        n_children: usize,
        mutation_strength: f32,
        map_buffer: Arc<Buffer<Affine>>,
        intermediate_buffers: impl IntoIterator<Item = wgpu::Buffer>,
        context: Context,
    ) -> impl Future<Output = ()> + 'static {
        let mut encoder = context
            .device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Evolution Scores Copying Command Encoder"),
            });
        for (idx, buf) in intermediate_buffers.into_iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                &buf,
                0,
                &self.scores,
                idx as u64 * mem::size_of::<u32>() as u64,
                mem::size_of::<u32>() as u64,
            );
        }
        context
            .queue()
            .submit(iter::once(encoder.finish()))
            .ignore();

        let maps = map_buffer.download(context.borrow());
        let scores = self.scores.download(context.borrow());
        let map_untyped_buffer: wgpu::Buffer = map_buffer.as_untyped().deref().clone();
        let context = context.into_static();

        async move {
            let (scores, maps) = futures::join!(scores, maps);
            let map_sets = maps
                .iter()
                .chunks(maps_per_set)
                .into_iter()
                .map(|chunk| chunk.collect_vec())
                .collect_vec();
            let mut indices = (0..scores.len())
                .flat_map(|idx| iter::repeat_n(idx, maps_per_set))
                .collect_vec();
            indices.sort_by_key(|&idx| cmp::Reverse(scores[idx]));
            let best_maps = indices
                .into_iter()
                .flat_map(|idx| &map_sets[idx])
                .copied()
                .copied()
                .take(elite_len * maps_per_set)
                .collect_vec();

            let mut children = best_maps.clone();
            let mut rng = rand::rng();
            for _ in 0..n_children {
                for map in &best_maps {
                    let AffineDecomposition {
                        angle,
                        shear,
                        scale:
                            Vec2 {
                                x: scalex,
                                y: scaley,
                            },
                        translation: Vec2 { x: trax, y: tray },
                    } = map.decomposition;

                    let mut noise = || rng.random_range(-mutation_strength..mutation_strength);

                    let child = AffineDecomposition {
                        angle: angle + noise() * f32::consts::TAU,
                        shear: shear + noise(),
                        // shear: 0.0,
                        scale: vec2(scalex + noise(), scaley + noise()),
                        translation: vec2(trax + noise(), tray + noise()),
                    }
                    .compose();
                    children.push(child);
                }
            }

            context
                .queue()
                .write_buffer(&map_untyped_buffer, 0, bytemuck::cast_slice(&children));
            context.queue().submit(iter::empty());
        }
    }
}
