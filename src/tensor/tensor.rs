use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use super::shape::{Broadcast, Same, SameNumElements, Shape, StaticShape, TRUE};
use super::layout::{Contiguous, Layout};
use super::slice_layout::SliceLayout;

#[derive(Debug, PartialEq)]
pub struct Tensor<T, S, L> {
    layout: L,
    _phantoms: PhantomData<(T, S)>,
}

impl<T, S, L> Default for Tensor<T, S, L>
where
    L: Default,
{
    fn default() -> Self {
        Tensor {
            layout: L::default(),
            _phantoms: PhantomData,
        }
    }
}

impl<'a, T, S> Tensor<T, S, SliceLayout<'a, T>> {
    pub fn from_slice(slice: &'a [T]) -> Self
    where
        S: StaticShape,
    {
        assert_eq!(
            slice.len(),
            S::NUM_ELEMENTS,
            "`slice` must have exactly {} elements to be be compatible with specified type-level shape. Got {}.",
            slice.len(),
            S::NUM_ELEMENTS
        );
        
        Tensor {
            layout: SliceLayout::from_slice_unchecked(slice, S::to_vec(), S::strides(), S::NUM_ELEMENTS, S::NUM_ELEMENTS),
            _phantoms: PhantomData,
        }
    }

    pub fn from_slice_dyn(slice: &'a [T], shape: Vec<usize>) -> Self
    where
        S: Shape,
    {
        assert!(S::runtime_compat(&shape), "`shape` is not compatible with specified type-level shape.");
        
        let mut num_elements = 1;
        let mut strides = Vec::with_capacity(shape.len());

        for dim in shape.iter() {
            strides.push(num_elements);
            num_elements *= dim;
        }

        Tensor {
            layout: SliceLayout::from_slice_unchecked(slice, shape, strides, num_elements, num_elements),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S, L> Tensor<T, S, L>
{
    pub fn broadcast<'a, Z>(&'a self) -> Tensor<T, Z, L::View>
    where
        S: StaticShape + Broadcast<Z>,
        Z: StaticShape,
        <S as Broadcast<Z>>::Output: TRUE,
        L: Layout<'a, T>,
    {
        let shape = Z::to_vec();
        let internal_strides = S::strides();
        let mut external_strides = self.layout.strides();
        external_strides.iter_mut().zip(internal_strides.iter()).zip(S::to_vec()).for_each(|((x, y), z)| {
            *x = if z == 1 {
                0
            } else {
                *x / *y
            }
        });
        
        let mut strides = Z::strides();
        let len = strides.len();
        strides.iter_mut().take(len - external_strides.len()).for_each(|x| *x = 0);
        strides.iter_mut().rev().zip(external_strides.iter().rev()).for_each(|(x, y)| *x *= *y);

        let opt_chunk_size = match strides.iter().rev().zip(Z::strides().into_iter().rev()).find(|(x, _)| **x == 0) {
            Some((_, y)) => y,
            None => Z::NUM_ELEMENTS,
        };
        
        Tensor {
            layout: self.layout.as_view_unchecked(shape, strides, Z::NUM_ELEMENTS, opt_chunk_size),
            _phantoms: PhantomData,
        }
    }

    pub fn reshape<'a, Z>(&'a self) -> Tensor<T, Z, L::View>
    where
        Z: StaticShape,
        S: SameNumElements<T, Z>,
        <S as SameNumElements<T, Z>>::Output: TRUE,
        L: Layout<'a, T> + Contiguous,
    {
        Tensor {
            layout: self.layout.as_view_unchecked(Z::to_vec(), Z::strides(), Z::NUM_ELEMENTS, Z::NUM_ELEMENTS),
            _phantoms: PhantomData,
        }
    }

    pub fn as_static<'a, Z>(&'a self) -> Tensor<T, Z, L::View>
    where
        Z: StaticShape,
        S: Same<Z>,
        <S as Same<Z>>::Output: TRUE,
        L: Layout<'a, T>,
    {
        Tensor {
            layout: self.layout.as_view_unchecked(Z::to_vec(), Z::strides(), Z::NUM_ELEMENTS, Z::NUM_ELEMENTS),
            _phantoms: PhantomData,
        }
    }

    pub fn as_view<'a>(&'a self) -> Tensor<T, S, L::View>
    where
        S: StaticShape,
        L: Layout<'a, T>,
    {
        Tensor {
            layout: self.layout.as_view_unchecked(S::to_vec(), S::strides(), S::NUM_ELEMENTS, S::NUM_ELEMENTS),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S, L> Deref for Tensor<T, S, L> {
    type Target = L;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl<T, S, L> DerefMut for Tensor<T, S, L>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.layout
    }
}