use super::layout::Layout;
use rayon::prelude::*;
use std::ops::Deref;

mod strided_chunks;
use strided_chunks::StridedChunks;

/// `SliceLayout` is a very flexible non-contiguous slice-backed layout.
/// It enables the creation of tensors that are views on other
/// tensors becuase it does not own the data. The internal slice
/// can actually point to any contiguous part of memory on the stack
/// or the heap that can be borrowed.
///
/// `SliceLayout` comes with some memory overhead to be able to keep
/// track of how borrowed data is used. It stores:
/// * the shape
/// * the actual strides (i.e. product of intrinsic and extrinsic strides)
/// * the number of elements
/// * the optimal chunk size (i.e. largest contiguous data pieces)
///
/// Note: The intrinsic strides are the cumulative product of the dimensions
/// and the extrinsic strides follow the ususal ML definition.
///
/// `SliceLayout` is the default view layout in Melange and should
/// be prefered unless you have specific needs.
#[derive(Debug, Clone)]
pub struct SliceLayout<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
    num_elements: usize,
    opt_chunk_size: usize,
}

impl<'a, T> SliceLayout<'a, T> {
    fn linear_index(&self, position: &[usize]) -> usize {
        position
            .iter()
            .rev()
            .zip(self.strides.iter().rev())
            .fold(0, |acc, (x, y)| acc + (x * y))
    }
    pub(super) fn chunk_at(&self, position: &[usize], size: usize) -> &[T] {
        let index = self.linear_index(position);
        &self.data[index..index + size]
    }

    pub fn from_slice_unchecked(
        slice: &'a [T],
        shape: Vec<usize>,
        strides: Vec<usize>,
        num_elements: usize,
        opt_chunk_size: usize,
    ) -> Self {
        SliceLayout {
            data: slice,
            shape,
            strides,
            num_elements,
            opt_chunk_size,
        }
    }
}

impl<'a, 'b, T> Layout<'b, T> for SliceLayout<'a, T>
where
    // T: 'b,
    T: 'static,
{
    type Iter = StridedChunks<'b, 'b, T>;
    type View = Self;
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
        self.opt_chunk_size
    }

    #[inline]
    fn num_elements(&self) -> usize {
        self.num_elements
    }

    #[inline]
    fn chunks(&'b self, chunk_size: usize) -> Self::Iter {
        StridedChunks::new(&self, chunk_size)
    }

    #[inline]
    fn as_view_unchecked(
        &'b self,
        shape: Vec<usize>,
        strides: Vec<usize>,
        num_elements: usize,
        opt_chunk_size: usize,
    ) -> Self::View {
        SliceLayout::from_slice_unchecked(&self.data, shape, strides, num_elements, opt_chunk_size)
    }
}

impl<'a, T> Deref for SliceLayout<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T> PartialEq for SliceLayout<'a, T>
where
    T: Send + Sync + PartialEq + 'static,
{
    fn eq(&self, other: &Self) -> bool {
        let chunk_size = self.opt_chunk_size.min(other.opt_chunk_size);

        if self.shape != other.shape {
            return false;
        }
        for (self_chunk, other_chunk) in self.chunks(chunk_size).zip(other.chunks(chunk_size)) {
            if !self_chunk
                .par_iter()
                .zip(other_chunk.par_iter())
                .all(|(x, y)| *x == *y)
            {
                return false;
            }
        }

        true
    }
}
