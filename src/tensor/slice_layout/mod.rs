use super::layout::Layout;
use rayon::prelude::*;
use std::ops::Deref;

mod strided_chunks;
use strided_chunks::StridedChunks;

#[derive(Debug)]
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
    T: 'b,
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
    T: Send + Sync + PartialEq,
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
