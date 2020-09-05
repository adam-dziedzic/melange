use std::ops::{Deref, DerefMut};
use super::layout::{Layout, LayoutMut, Alloc, Contiguous, OpsDefaultOutput, OpsAllocOutput};
use super::slice_layout::SliceLayout;
use super::stack_layout::StackLayout;
use super::shape::{StaticShape, NumElements};

pub struct HeapLayout<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> Alloc for HeapLayout<T>
where
    T: Default + Clone,
{
    fn alloc(shape: Vec<usize>) -> Self {
        let mut num_elements = 1;
        let mut strides = shape.clone();

        for stride in strides.iter_mut().rev() {
            let tmp = num_elements;
            num_elements *= *stride;
            *stride = tmp;
        }
        
        HeapLayout {
            data: vec![T::default(); num_elements],
            shape,
            strides,
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
        self.data.len()
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

impl<'a, T, S> OpsDefaultOutput<T, S> for HeapLayout<T>
where
    S: StaticShape + NumElements<T>,
    T: Default + 'static,
{
    type Default = StackLayout<T, S>;
}

impl<'a, T> OpsAllocOutput<T> for HeapLayout<T>
where
    T: Default + Clone + 'static,
{
    type Alloc = Self;
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