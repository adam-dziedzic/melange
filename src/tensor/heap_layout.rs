use std::ops::{Deref, DerefMut};
use super::layout::{Layout, LayoutMut, Contiguous};
use super::slice_layout::SliceLayout;

pub struct HeapLayout<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    num_elements: usize,
    opt_chunk_size: usize,
}

impl<T> Default for HeapLayout<T>
where
    T: Default,
{
    fn default() -> Self {
        HeapLayout {
            
        }
    }
}

impl<'a, T> Layout<'a, T> for HeapLayout<T>
where
    T: 'a,
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
        self.opt_chunk_size
    }

    #[inline]
    fn chunks(&'a self, chunk_size: usize) -> Self::Iter {
        self.data.chunks(chunk_size)
    }

    #[inline]
    fn as_view_unchecked(&'a self, shape: Vec<usize>, strides: Vec<usize>, num_elements: usize, opt_chunk_size: usize) -> Self::View {        
        SliceLayout::from_slice_unchecked(&self.data, shape, strides, num_elements, opt_chunk_size)
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

impl<T> Contiguous for HeapLayout<T> {}

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
