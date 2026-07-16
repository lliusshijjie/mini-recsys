//! Aligned buffers for hot primitive arrays.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

const CACHE_LINE_ALIGNMENT: usize = 64;

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
}

pub(crate) trait ZeroedPrimitive: Copy + sealed::Sealed {}

impl ZeroedPrimitive for u8 {}
impl ZeroedPrimitive for u32 {}
impl ZeroedPrimitive for f32 {}

#[derive(Debug)]
pub(crate) struct AlignedBuffer<T: ZeroedPrimitive> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    _marker: PhantomData<T>,
}

impl<T: ZeroedPrimitive> AlignedBuffer<T> {
    pub(crate) fn new_zeroed(len: usize) -> Self {
        let element_size = mem::size_of::<T>();
        let size = len
            .checked_mul(element_size)
            .expect("aligned buffer size overflow");
        let layout = Layout::from_size_align(size.max(1), CACHE_LINE_ALIGNMENT)
            .expect("valid aligned buffer layout");
        let raw_ptr = unsafe { alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(raw_ptr).unwrap_or_else(|| handle_alloc_error(layout));
        Self {
            ptr,
            len,
            layout,
            _marker: PhantomData,
        }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[cfg(test)]
    pub(crate) fn ptr_alignment(&self) -> usize {
        self.ptr.as_ptr() as usize
    }
}

impl<T: ZeroedPrimitive> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

impl<T: ZeroedPrimitive> Deref for AlignedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: ZeroedPrimitive> DerefMut for AlignedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}
