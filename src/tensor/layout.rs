pub trait Layout<'a, T>
where
    T: 'a
{
    type Iter: Iterator<Item = &'a [T]>;
    type View: Layout<'a, T>;

    fn shape(&self) -> Vec<usize>;
    fn strides(&self) -> Vec<usize>;
    fn opt_chunk_size(&self) -> usize;
    fn chunks(&'a self, chunk_size: usize) -> Self::Iter;
    fn as_view_unchecked(&'a self, shape: Vec<usize>, strides: Vec<usize>, num_elements: usize, opt_chunk_size: usize) -> Self::View;
}

pub trait LayoutMut<'a, T>
where
    T: 'a
{
    type IterMut: Iterator<Item = &'a mut [T]>;

    fn chunks_mut(&'a mut self, chunk_size: usize) -> Self::IterMut;
}

pub trait OpsDefaultOutput<T, S>
{
    type Default: for<'a> LayoutMut<'a, T> + Default;
}

pub trait OpsAllocOutput<T>
{
    type Alloc: for<'a> LayoutMut<'a, T> + Alloc;
}

pub trait Alloc {
    fn alloc(shape: Vec<usize>) -> Self;
}

pub trait Contiguous {}