use std::{marker::PhantomData, mem, ops::Deref};

use bytemuck::Pod;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages,
};

use crate::app::Context;

pub type UntypedBuffer = wgpu::Buffer;

#[derive(Debug)]
pub struct Buffer<T: Pod> {
    buffer: UntypedBuffer,
    _marker: PhantomData<T>,
}

impl<T: Pod> Buffer<T> {
    pub fn new(data: &[T], label: Option<&str>, usage: BufferUsages, context: Context) -> Self {
        let bytes = bytemuck::cast_slice(data);
        assert!(u32::try_from(bytes.len()).is_ok());
        let buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label,
            contents: bytes,
            usage,
        });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.size() as usize / mem::size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len_u32(&self) -> u32 {
        self.len() as u32
    }

    pub fn as_untyped(&self) -> &UntypedBuffer {
        &self.buffer
    }
}

impl<T: Pod> Deref for Buffer<T> {
    type Target = UntypedBuffer;
    fn deref(&self) -> &Self::Target {
        self.as_untyped()
    }
}
