use std::{future::Future, iter, marker::PhantomData, mem, ops::Deref};

use bytemuck::Pod;
use wgpu::{util::BufferInitDescriptor, BufferDescriptor, BufferUsages, CommandEncoderDescriptor};
use wgpu_async::AsyncBuffer;

use crate::app::Context;

pub type UntypedBuffer = AsyncBuffer;

#[derive(Debug)]
pub struct Buffer<T: Pod> {
    untyped: UntypedBuffer,
    _marker: PhantomData<T>,
}

impl<T: Pod> Buffer<T> {
    pub fn new(len: usize, label: Option<&str>, usage: BufferUsages, context: Context) -> Self {
        let untyped = context.device().create_buffer(&BufferDescriptor {
            label,
            size: (len * mem::size_of::<T>()) as u64,
            usage,
            mapped_at_creation: false,
        });

        Self {
            untyped,
            _marker: PhantomData,
        }
    }

    pub fn from_data(
        data: &[T],
        label: Option<&str>,
        usage: BufferUsages,
        context: Context,
    ) -> Self {
        let bytes = bytemuck::cast_slice(data);
        assert!(u32::try_from(bytes.len()).is_ok());
        let untyped = context.device().create_buffer_init(&BufferInitDescriptor {
            label,
            contents: bytes,
            usage,
        });

        Self {
            untyped,
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.untyped.size() as usize / mem::size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len_u32(&self) -> u32 {
        self.len() as u32
    }

    pub fn as_untyped(&self) -> &UntypedBuffer {
        &self.untyped
    }

    pub fn download(&self, context: Context) -> impl Future<Output = Vec<T>> + 'static {
        let context = context.into_static();

        let read_buffer = context.device().create_buffer(&BufferDescriptor {
            label: None,
            size: self.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut copy_encoder = context
            .device()
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        copy_encoder.copy_buffer_to_buffer(self, 0, &read_buffer, 0, self.size());
        let copy_command = copy_encoder.finish();

        async move {
            context.queue().submit(iter::once(copy_command)).await;

            let slice = read_buffer.slice(..);
            slice
                .map_async(wgpu::MapMode::Read)
                .await
                .expect("failed to map buffer");
            let map_data = {
                let mapped_range = slice.get_mapped_range();
                bytemuck::cast_slice(&mapped_range).to_vec()
            };
            read_buffer.unmap();
            map_data
        }
    }
}

impl<T: Pod> Deref for Buffer<T> {
    type Target = UntypedBuffer;
    fn deref(&self) -> &Self::Target {
        self.as_untyped()
    }
}

impl<T: Pod> AsRef<Buffer<T>> for Buffer<T> {
    fn as_ref(&self) -> &Buffer<T> {
        self
    }
}
