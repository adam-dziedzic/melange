use super::layout::{Alloc, Layout};
use super::shape::{
    internal_strides_in_place, Broadcast, Same, SameNumElements, Shape, StaticShape, StridedShape,
    StridedShapeDyn, TRUE,
};
use super::slice_layout::SliceLayout;
use super::transpose_policy::{Contiguous, Strided};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

#[derive(Debug, PartialEq)]
pub struct Tensor<T, S, C, L, P> {
    layout: L,
    _phantoms: PhantomData<(T, S, C, P)>,
}

impl<T, S, C, L, P> Default for Tensor<T, S, C, L, P>
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

impl<'a, T, S, C, P> Tensor<T, S, C, SliceLayout<'a, T>, P> {
    pub fn from_slice(slice: &'a [T]) -> Self
    where
        S: StaticShape,
    {
        assert_eq!(
            S::NUM_ELEMENTS,
            slice.len(),
            "`slice` must have exactly {} elements to be be compatible with specified type-level shape. Got {}.",
            S::NUM_ELEMENTS,
            slice.len(),
        );
        Tensor {
            layout: SliceLayout::from_slice_unchecked(
                slice,
                S::to_vec(),
                S::strides(),
                S::NUM_ELEMENTS,
                S::NUM_ELEMENTS,
            ),
            _phantoms: PhantomData,
        }
    }

    pub fn from_slice_dyn(slice: &'a [T], shape: Vec<usize>) -> Self
    where
        S: Shape,
    {
        assert!(
            S::runtime_compat(&shape),
            "`shape` is not compatible with specified type-level shape."
        );
        let mut num_elements = 1;
        let mut strides = Vec::with_capacity(shape.len());

        for dim in shape.iter() {
            strides.push(num_elements);
            num_elements *= dim;
        }

        Tensor {
            layout: SliceLayout::from_slice_unchecked(
                slice,
                shape,
                strides,
                num_elements,
                num_elements,
            ),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S, C, L, P> Tensor<T, S, C, L, P> {
    pub fn broadcast<'a, Z>(&'a self) -> Tensor<T, Z, Strided, L::View, P>
    where
        S: StaticShape + Broadcast<Z>,
        Z: StaticShape,
        <S as Broadcast<Z>>::Output: TRUE,
        L: Layout<'a, T>,
    {
        let shape = Z::to_vec();
        let internal_strides = S::strides();
        let mut external_strides = self.strides();
        external_strides
            .iter_mut()
            .zip(internal_strides.iter())
            .zip(S::to_vec())
            .for_each(|((x, y), z)| *x = if z == 1 { 0 } else { *x / *y });
        let mut strides = Z::strides();
        let len = strides.len();
        strides
            .iter_mut()
            .take(len - external_strides.len())
            .for_each(|x| *x = 0);
        strides
            .iter_mut()
            .rev()
            .zip(external_strides.iter().rev())
            .for_each(|(x, y)| *x *= *y);

        let opt_chunk_size = match strides
            .iter()
            .rev()
            .zip(Z::strides().into_iter().rev())
            .find(|(x, _)| **x == 0)
        {
            Some((_, y)) => y,
            None => Z::NUM_ELEMENTS,
        };
        Tensor {
            layout: self.as_view_unchecked(shape, strides, Z::NUM_ELEMENTS, opt_chunk_size),
            _phantoms: PhantomData,
        }
    }

    pub fn broadcast_dynamic<'a, Z>(
        &'a self,
        shape: Vec<usize>,
    ) -> Tensor<T, Z, Strided, L::View, P>
    where
        S: Broadcast<Z>,
        <S as Broadcast<Z>>::Output: TRUE,
        L: Layout<'a, T>,
    {
        let current_shape = self.shape();
        assert!(
            shape
                .iter()
                .rev()
                .zip(current_shape.iter().rev())
                .all(|(x, y)| *x == 1 || *y == 1 || *x == *y),
            "Cannot broadcast shapes {:?} and {:?}.",
            current_shape,
            shape,
        );
        let internal_strides = internal_strides_in_place(current_shape);
        let mut external_strides = self.strides();
        external_strides
            .iter_mut()
            .zip(internal_strides.iter())
            .zip(self.shape())
            .for_each(|((x, y), z)| *x = if z == 1 { 0 } else { *x / *y });
        let mut strides = internal_strides_in_place(shape.clone());
        let len = strides.len();
        strides
            .iter_mut()
            .take(len - external_strides.len())
            .for_each(|x| *x = 0);
        strides
            .iter_mut()
            .rev()
            .zip(external_strides.iter().rev())
            .for_each(|(x, y)| *x *= *y);

        let num_elements = shape.iter().product();

        let opt_chunk_size = match strides
            .iter()
            .rev()
            .zip(strides.clone().into_iter().rev())
            .find(|(x, _)| **x == 0)
        {
            Some((_, y)) => y,
            None => num_elements,
        };
        Tensor {
            layout: self.as_view_unchecked(shape, strides, num_elements, opt_chunk_size),
            _phantoms: PhantomData,
        }
    }

    pub fn stride<'a, Z>(&'a self) -> Tensor<T, <S as StridedShape<Z>>::Output, Strided, L::View, P>
    where
        S: StaticShape + StridedShape<Z>,
        Z: StaticShape,
        L: Layout<'a, T>,
    {
        let mut strides = self.strides();
        strides
            .iter_mut()
            .zip(Z::to_vec())
            .for_each(|(x, y)| *x *= y);

        let opt_chunk_size = strides
            .iter()
            .zip(S::to_vec())
            .take_while(|(x, _)| **x == 1)
            .map(|(_, y)| y)
            .product();
        Tensor {
            layout: self.as_view_unchecked(
                <S as StridedShape<Z>>::Output::to_vec(),
                strides,
                <S as StridedShape<Z>>::Output::NUM_ELEMENTS,
                opt_chunk_size,
            ),
            _phantoms: PhantomData,
        }
    }

    pub fn stride_dynamic<'a, Z>(
        &'a self,
        strides: Vec<usize>,
    ) -> Tensor<T, <S as StridedShapeDyn<Z>>::Output, Strided, L::View, P>
    where
        S: StridedShapeDyn<Z>,
        L: Layout<'a, T>,
    {
        let external_strides = strides;
        let mut strides = self.strides();
        strides
            .iter_mut()
            .zip(external_strides.iter())
            .for_each(|(x, y)| *x *= y);

        let mut shape = self.shape();
        shape
            .iter_mut()
            .zip(strides.iter())
            .for_each(|(x, y)| *x = *x / *y + (*x % *y).min(1));
        let num_elements = shape.iter().product();

        let opt_chunk_size = strides
            .iter()
            .zip(strides.clone().into_iter())
            .take_while(|(x, _)| **x == 1)
            .map(|(_, y)| y)
            .product();
        Tensor {
            layout: self.as_view_unchecked(shape, strides, num_elements, opt_chunk_size),
            _phantoms: PhantomData,
        }
    }

    pub fn as_static<'a, Z>(&'a self) -> Tensor<T, Z, C, L::View, P>
    where
        Z: StaticShape,
        S: Same<Z>,
        <S as Same<Z>>::Output: TRUE,
        L: Layout<'a, T>,
    {
        Tensor {
            layout: self.as_view_unchecked(
                Z::to_vec(),
                Z::strides(),
                Z::NUM_ELEMENTS,
                Z::NUM_ELEMENTS,
            ),
            _phantoms: PhantomData,
        }
    }

    pub fn as_view<'a>(&'a self) -> Tensor<T, S, C, L::View, P>
    where
        S: StaticShape,
        L: Layout<'a, T>,
    {
        Tensor {
            layout: self.as_view_unchecked(
                S::to_vec(),
                S::strides(),
                S::NUM_ELEMENTS,
                S::NUM_ELEMENTS,
            ),
            _phantoms: PhantomData,
        }
    }

    pub fn alloc(shape: Vec<usize>) -> Self
    where
        L: Alloc,
    {
        Tensor {
            layout: L::alloc(shape),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S, L, P> Tensor<T, S, Contiguous, L, P> {
    pub fn reshape<'a, Z>(&'a self) -> Tensor<T, Z, Contiguous, L::View, P>
    where
        Z: StaticShape,
        S: SameNumElements<T, Z>,
        <S as SameNumElements<T, Z>>::Output: TRUE,
        L: Layout<'a, T>,
    {
        Tensor {
            layout: self.as_view_unchecked(
                Z::to_vec(),
                Z::strides(),
                Z::NUM_ELEMENTS,
                Z::NUM_ELEMENTS,
            ),
            _phantoms: PhantomData,
        }
    }

    pub fn reshape_dynamic<'a, Z>(
        &'a self,
        shape: Vec<usize>,
    ) -> Tensor<T, Z, Contiguous, L::View, P>
    where
        L: Layout<'a, T>,
    {
        let num_elements = shape.iter().product();
        let current_shape = self.shape();
        assert_eq!(
            num_elements,
            current_shape.iter().product(),
            "Cannot reshape Tensor of shape {:?} to {:?}. Differing number of elements.",
            current_shape,
            shape,
        );
        let strides = internal_strides_in_place(shape.clone());

        Tensor {
            layout: self.as_view_unchecked(shape, strides, num_elements, num_elements),
            _phantoms: PhantomData,
        }
    }

    // pub fn transpose<'a, Z>(&'a self) -> Tensor<T, Z, Contiguous, L::View, P>
    // where
    //     Z: StaticShape,
    //     S: SameNumElements<T, Z>,
    //     <S as SameNumElements<T, Z>>::Output: TRUE,
    //     L: Layout<'a, T>,
    // {
    //     Tensor {
    //         layout: self.as_view_unchecked(
    //             Z::to_vec(),
    //             Z::strides(),
    //             Z::NUM_ELEMENTS,
    //             Z::NUM_ELEMENTS,
    //         ),
    //         _phantoms: PhantomData,
    //     }
    // }
}

impl<T, S, C, L, P> Deref for Tensor<T, S, C, L, P> {
    type Target = L;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl<T, S, C, L, P> DerefMut for Tensor<T, S, C, L, P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.layout
    }
}
