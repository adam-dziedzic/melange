use super::layout::{Layout, LayoutMut};
use super::shape::{NumElements, StaticShape};
use super::slice_layout::SliceLayout;
use generic_array::GenericArray;
use std::ops::{Deref, DerefMut};

#[derive(Debug, PartialEq)]
pub struct StackLayout<T, S>
where
    S: NumElements<T>,
{
    data: GenericArray<T, <S as NumElements<T>>::Output>,
}

impl<T, S> Default for StackLayout<T, S>
where
    S: NumElements<T>,
    T: Default,
{
    fn default() -> Self {
        StackLayout {
            data: GenericArray::default(),
        }
    }
}

impl<'a, T, S> Layout<'a, T> for StackLayout<T, S>
where
    S: StaticShape + NumElements<T>,
    <S as NumElements<T>>::Output: 'static,
    T: 'a,
{
    type Iter = std::slice::Chunks<'a, T>;
    type View = SliceLayout<'a, T>;

    #[inline]
    fn shape(&self) -> Vec<usize> {
        S::to_vec()
    }

    #[inline]
    fn strides(&self) -> Vec<usize> {
        S::to_vec()
    }

    #[inline]
    fn opt_chunk_size(&self) -> usize {
        S::NUM_ELEMENTS
    }

    #[inline]
    fn num_elements(&self) -> usize {
        S::NUM_ELEMENTS
    }

    #[inline]
    fn chunks(&'a self, chunk_size: usize) -> Self::Iter {
        self.data.as_slice().chunks(chunk_size)
    }

    fn as_view_unchecked(
        &'a self,
        shape: Vec<usize>,
        strides: Vec<usize>,
        num_elements: usize,
        opt_chunk_size: usize,
    ) -> Self::View {
        SliceLayout::from_slice_unchecked(&self.data, shape, strides, num_elements, opt_chunk_size)
    }
}

impl<'a, T, S> LayoutMut<'a, T> for StackLayout<T, S>
where
    S: StaticShape + NumElements<T>,
    <S as NumElements<T>>::Output: 'static,
    T: 'a,
{
    type IterMut = std::slice::ChunksMut<'a, T>;

    #[inline]
    fn chunks_mut(&'a mut self, chunk_size: usize) -> Self::IterMut {
        self.data.as_mut_slice().chunks_mut(chunk_size)
    }
}

impl<T, S> Deref for StackLayout<T, S>
where
    S: NumElements<T>,
    <S as NumElements<T>>::Output: 'static,
{
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.data.as_slice()
    }
}

impl<T, S> DerefMut for StackLayout<T, S>
where
    S: NumElements<T>,
    <S as NumElements<T>>::Output: 'static,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut_slice()
    }
}
