use super::layout::{Alloc, DynamicFill, Layout, StaticFill};
use super::shape::{
    intrinsic_strides_in_place, Broadcast, Same, SameNumElements, StaticShape, StridedShape,
    StridedShapeDyn, Transpose, TRUE, Static, Dynamic, Reduction,
};
use super::transpose_policy::{Contiguous, Strided, TransposePolicy};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::convert::TryFrom;
use super::heap_layout::HeapLayout;
use std::io::{Error, ErrorKind};
use super::static_heap_layout::StaticHeapLayout;
use typenum::U0;

/// The central struct of the `tensor` module.
///
/// `Tensor` is highly generic structure that provides a unique interface
/// for all combinations of shapes, layouts and policies.
/// It is parametrized by the following generics:
/// * T the data scalar type,
/// * S the type-level shape for compile time checks,
/// * D a type level bit, true if the tensor is dynamically shaped,
/// * C the transpose policy
/// * L the layout
/// * P the allocation policy
///
/// The behavior of the tensor is fully determined by those type
/// parameters. Although the roles of T and S are easy to understand,
/// the other parameters might seem more obscure.
///
/// C is primarily used with BLAS-backed operation that require
/// the data to be layed out in a contiguous manner and in a specific order.
/// Two flag are available in the contiguous case: `Contiguous` and `Transposed`.
/// `Contiguous` is the default case, `Transposed` is used as a flag for tensors
/// whose axes has been inverted. Although axes inversion is sufficient for most
/// Melange operations to work, it is not the case with BLAS.
/// The `Strided` flag is used for tensors that are not contiguous and shouldn't
/// be passed to BLAS operations (which is enforced at compile time).
///
/// L represents the Layout that internally stores the tensor's data, there are various
/// Layout with various properties but the default are a good choice for most cases:
/// * `StaticHeapLayout` with static shapes,
/// * `HeapLayout` with dynamic shapes,
/// * `SliceLayout` for views.
///
/// P is the policy that is used when a new tensor needs to be allocated
/// such as the resul of an operation on borrowed tensors. Note that by convention,
/// operations use and pass `self`'s policy. `DefaultPolicy` allocates with the defaults
/// suggested in the previous paragraph.
///
/// For ease of use, aliases for common cases are defined in the `prelude` of the `tensor` module.
#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T, S, D, C, L, P> {
    layout: L,
    _phantoms: PhantomData<(T, S, D, C, P)>,
}

impl<T, S, D, C, L, P> Default for Tensor<T, S, D, C, L, P>
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

impl<T, S, P> TryFrom<Vec<T>> for Tensor<T, S, Static, Contiguous, StaticHeapLayout<T, S>, P>
where
    S: StaticShape,
{
    type Error = std::io::Error;
    
    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        if S::NUM_ELEMENTS != v.len() {
            return Err(Error::new(ErrorKind::InvalidData, format!("Given vector is not compatible with type-level shape: got {} elements instead of {}", v.len(), S::NUM_ELEMENTS)))
        }
        
        Ok(Tensor {
            layout: StaticHeapLayout {
                data: v,
                _phantoms: PhantomData,
            },
            _phantoms: PhantomData,
        })
    }
}

impl<T, S, P> TryFrom<Vec<T>> for Tensor<T, S, Dynamic, Contiguous, HeapLayout<T>, P>
where
    S: Reduction<U0>,
    <S as Reduction<U0>>::Output: StaticShape,
{
    type Error = std::io::Error;
    
    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        if v.len() % <S as Reduction<U0>>::Output::NUM_ELEMENTS != 0 {
            return Err(Error::new(ErrorKind::InvalidData, format!("Given vector is not compatible with type-level shape: got {} elements instead of a multiple of {}", v.len(), <S as Reduction<U0>>::Output::NUM_ELEMENTS)))
        }

        let mut shape = <S as Reduction<U0>>::Output::to_vec();
        shape[0] = v.len() / <S as Reduction<U0>>::Output::NUM_ELEMENTS;
        let strides = <S as Reduction<U0>>::Output::strides();

        Ok(Tensor {
            layout: HeapLayout {
                data: v,
                shape,
                strides,
            },
            _phantoms: PhantomData,
        })
    }
}

// impl<T, P> From<Vec<T>> for Tensor<T, Shape1D<Dyn>, Dynamic, Contiguous, HeapLayout<T>, P> {
//     fn from(data: Vec<T>) -> Self {
//         let len = data.len();
//         Tensor {
//             layout: HeapLayout {
//                 data,
//                 shape: vec![len],
//                 strides: vec![1],
//             },
//             _phantoms: PhantomData,
//         }
//     }
// }

impl<T, S, C, L, P> Tensor<T, S, Static, C, L, P> {
    pub fn broadcast<Z>(&self) -> Tensor<T, Z, Static, Strided, <L as Layout<'_, T>>::View, P>
    where
        S: StaticShape + Broadcast<Z>,
        Z: StaticShape,
        <S as Broadcast<Z>>::Output: TRUE,
        L: for<'a> Layout<'a, T>,
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

    pub fn stride<Z>(
        &self,
    ) -> Tensor<T, <S as StridedShape<Z>>::Output, Static, Strided, <L as Layout<'_, T>>::View, P>
    where
        S: StaticShape + StridedShape<Z>,
        Z: StaticShape,
        L: for<'a> Layout<'a, T>,
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

    pub fn fill(value: T) -> Self
    where
        L: StaticFill<T>,
    {
        Tensor {
            layout: L::fill(value),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S, C, L, P> Tensor<T, S, Dynamic, C, L, P> {
    pub fn broadcast<Z>(
        &self,
        shape: Vec<usize>,
    ) -> Tensor<T, Z, Dynamic, Strided, <L as Layout<'_, T>>::View, P>
    where
        S: Broadcast<Z>,
        <S as Broadcast<Z>>::Output: TRUE,
        L: for<'a> Layout<'a, T>,
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
        let internal_strides = intrinsic_strides_in_place(current_shape);
        let mut external_strides = self.strides();
        external_strides
            .iter_mut()
            .zip(internal_strides.iter())
            .zip(self.shape())
            .for_each(|((x, y), z)| *x = if z == 1 { 0 } else { *x / *y });
        let mut strides = intrinsic_strides_in_place(shape.clone());
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

    pub fn stride<Z>(
        &self,
        strides: Vec<usize>,
    ) -> Tensor<T, <S as StridedShapeDyn<Z>>::Output, Dynamic, Strided, <L as Layout<'_, T>>::View, P>
    where
        S: StridedShapeDyn<Z>,
        L: for<'a> Layout<'a, T>,
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

    pub fn as_static<Z>(&self) -> Tensor<T, Z, Static, C, <L as Layout<'_, T>>::View, P>
    where
        Z: StaticShape,
        S: Same<Z>,
        <S as Same<Z>>::Output: TRUE,
        L: for<'a> Layout<'a, T>,
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

    pub fn fill(value: T, shape: Vec<usize>) -> Self
    where
        L: DynamicFill<T>,
    {
        Tensor {
            layout: L::fill(value, shape),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    pub fn transpose(
        &self,
    ) -> Tensor<T, <S as Transpose>::Output, D, C::Transposed, <L as Layout<'_, T>>::View, P>
    where
        S: Transpose,
        C: TransposePolicy,
        L: for<'a> Layout<'a, T>,
    {
        Tensor {
            layout: self.as_view_unchecked(
                self.shape().into_iter().rev().collect(),
                self.strides().into_iter().rev().collect(),
                self.num_elements(),
                1,
            ),
            _phantoms: PhantomData,
        }
    }

    pub fn as_view(&self) -> Tensor<T, S, D, C, <L as Layout<'_, T>>::View, P>
    where
        S: StaticShape,
        L: for<'a> Layout<'a, T>,
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

impl<T, S, L, P> Tensor<T, S, Static, Contiguous, L, P> {
    pub fn reshape<Z>(&self) -> Tensor<T, Z, Static, Contiguous, <L as Layout<'_, T>>::View, P>
    where
        Z: StaticShape,
        S: SameNumElements<T, Z>,
        <S as SameNumElements<T, Z>>::Output: TRUE,
        L: for<'a> Layout<'a, T>,
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
}

impl<T, S, L, P> Tensor<T, S, Dynamic, Contiguous, L, P> {
    pub fn reshape<Z>(
        &self,
        shape: Vec<usize>,
    ) -> Tensor<T, Z, Dynamic, Contiguous, <L as Layout<'_, T>>::View, P>
    where
        L: for<'a> Layout<'a, T>,
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
        let strides = intrinsic_strides_in_place(shape.clone());

        Tensor {
            layout: self.as_view_unchecked(shape, strides, num_elements, num_elements),
            _phantoms: PhantomData,
        }
    }
}

impl<T, S, D, C, L, P> Deref for Tensor<T, S, D, C, L, P> {
    type Target = L;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl<T, S, D, C, L, P> DerefMut for Tensor<T, S, D, C, L, P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.layout
    }
}
