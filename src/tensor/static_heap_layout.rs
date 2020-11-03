use super::layout::{Layout, LayoutMut, StaticFill};
use super::shape::StaticShape;
use super::slice_layout::SliceLayout;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// `Vec`-backed contiguous layout that does require a shape known
/// at compile time. This comes with no memory overhead since both
/// the shape and strides are encoded in the type and checked at compile time.
///
/// `StaticHeapLayout` is the default static storage in Melange and should
/// be prefered unless you have specific needs.
#[derive(Debug, PartialEq, Clone)]
pub struct StaticHeapLayout<T, S> {
    pub(super) data: Vec<T>,
    pub(super) _phantoms: PhantomData<S>,
}

impl<T, S> Default for StaticHeapLayout<T, S>
where
    T: Default + Clone,
    S: StaticShape,
{
    fn default() -> Self {
        StaticHeapLayout {
            data: vec![T::default(); S::NUM_ELEMENTS],
            _phantoms: PhantomData,
        }
    }
}

impl<T, S> StaticFill<T> for StaticHeapLayout<T, S>
where
    T: Clone,
    S: StaticShape,
{
    fn fill(value: T) -> Self {
        StaticHeapLayout {
            data: vec![value; S::NUM_ELEMENTS],
            _phantoms: PhantomData,
        }
    }
}

impl<'a, T, S> Layout<'a, T> for StaticHeapLayout<T, S>
where
    // T: 'a,
    T: 'static,
    S: StaticShape,
{
    type Iter = std::slice::Chunks<'a, T>;
    type View = SliceLayout<'a, T>;
    #[inline]
    fn shape(&self) -> Vec<usize> {
        S::to_vec()
    }

    #[inline]
    fn strides(&self) -> Vec<usize> {
        S::strides()
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

impl<'a, T, S> LayoutMut<'a, T> for StaticHeapLayout<T, S>
where
    T: 'a,
{
    type IterMut = std::slice::ChunksMut<'a, T>;

    #[inline]
    fn chunks_mut(&'a mut self, chunk_size: usize) -> Self::IterMut {
        self.data.as_mut_slice().chunks_mut(chunk_size)
    }
}

impl<T, S> Deref for StaticHeapLayout<T, S> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.data.as_slice()
    }
}

impl<T, S> DerefMut for StaticHeapLayout<T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut_slice()
    }
}
