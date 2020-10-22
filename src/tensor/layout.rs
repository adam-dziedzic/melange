use std::ops::{Deref, DerefMut};

/// This trait defines the basic behavior of any data layout.
///
/// It provides utility methods to access basic information
/// about the data such as its shape and strides.
///
/// It also provides a `chunks` method that returns an Iterator
/// of `&[T]` slices of a given length whose type is defined by the
/// implementation.
/// This method is virtually used by all the mathematical operations
/// to get maximal contiguous chunks of data and optimize parallelization.
///
/// `Layout` also defines an associated  type `View` that represents the
/// Layout that should be used by all non-allocating operations
/// (e.g. broadcasting).
pub trait Layout<'a, T>: Deref<Target = [T]>
where
    T: 'a,
{
    type Iter: Iterator<Item = &'a [T]>;
    type View: for<'b> Layout<'b, T>;

    fn shape(&self) -> Vec<usize>;
    fn strides(&self) -> Vec<usize>;
    fn opt_chunk_size(&self) -> usize;
    fn num_elements(&self) -> usize;
    fn chunks(&'a self, chunk_size: usize) -> Self::Iter;
    fn as_view_unchecked(
        &'a self,
        shape: Vec<usize>,
        strides: Vec<usize>,
        num_elements: usize,
        opt_chunk_size: usize,
    ) -> Self::View;
}

pub trait LayoutMut<'a, T>: DerefMut<Target = [T]>
where
    T: 'a,
{
    type IterMut: Iterator<Item = &'a mut [T]>;

    fn chunks_mut(&'a mut self, chunk_size: usize) -> Self::IterMut;
}

pub trait Alloc {
    fn alloc(shape: Vec<usize>) -> Self;
}

pub trait StaticFill<T> {
    fn fill(value: T) -> Self;
}

pub trait DynamicFill<T> {
    fn fill(value: T, shape: Vec<usize>) -> Self;
}
