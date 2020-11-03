use super::layout::{Alloc, DynamicFill, Layout, LayoutMut};
use super::slice_layout::SliceLayout;
use std::ops::{Deref, DerefMut};

/// `Vec`-backed contiguous layout that does not require a shape known
/// at compile time. This comes with some memory overhead to store both
/// the shape and intrinsic strides (i.e. cumulative products of the
/// dimensions).
///
/// `HeapLayout` is the default dynamic storage in Melange and should
/// be prefered unless you have specific needs.
#[derive(Debug, PartialEq, Clone)]
pub struct HeapLayout<T> {
    pub(super) data: Vec<T>,
    pub(super) shape: Vec<usize>,
    pub(super) strides: Vec<usize>,
}

impl<T> Alloc for HeapLayout<T>
where
    T: Default + Clone,
{
    #[inline]
    fn alloc(shape: Vec<usize>) -> Self {
        Self::fill(T::default(), shape)
    }
}

impl<T> DynamicFill<T> for HeapLayout<T>
where
    T: Clone,
{
    fn fill(value: T, shape: Vec<usize>) -> Self {
        let mut num_elements = 1;
        let mut strides = shape.clone();

        for stride in strides.iter_mut().rev() {
            let tmp = num_elements;
            num_elements *= *stride;
            *stride = tmp;
        }
        HeapLayout {
            data: vec![value; num_elements],
            shape,
            strides,
        }
    }
}

impl<'a, T> Layout<'a, T> for HeapLayout<T>
where
    // T: 'a,
    T: 'static,
{
    type Iter = std::slice::Chunks<'a, T>;
    type View = SliceLayout<'a, T>;
    #[inline]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[inline]
    fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }
    #[inline]
    fn opt_chunk_size(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn num_elements(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn chunks(&'a self, chunk_size: usize) -> Self::Iter {
        self.data.chunks(chunk_size)
    }

    #[inline]
    fn as_view_unchecked(
        &'a self,
        shape: Vec<usize>,
        strides: Vec<usize>,
        num_elements: usize,
        opt_chunk_size: usize,
    ) -> Self::View {
        SliceLayout {
            data: &self.data,
            shape,
            strides,
            num_elements,
            opt_chunk_size,
        }
    }
}

impl<'a, T> LayoutMut<'a, T> for HeapLayout<T>
where
    T: 'a,
{
    type IterMut = std::slice::ChunksMut<'a, T>;

    #[inline]
    fn chunks_mut(&'a mut self, chunk_size: usize) -> Self::IterMut {
        self.data.as_mut_slice().chunks_mut(chunk_size)
    }
}

impl<T> Deref for HeapLayout<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.data.as_slice()
    }
}

impl<T> DerefMut for HeapLayout<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut_slice()
    }
}
