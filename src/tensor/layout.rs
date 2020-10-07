use std::ops::{Deref, DerefMut};

pub trait Layout<'a, T>: Deref<Target = [T]>
where
    T: 'a,
{
    type Iter: Iterator<Item = &'a [T]>;
    type View: Layout<'a, T>;

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
