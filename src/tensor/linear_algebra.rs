extern crate cblas;
extern crate openblas_src;

use super::allocation_policy::{DynamicAllocationPolicy, StaticAllocationPolicy};
use super::layout::Layout;
use super::shape::{Shape2D, TRUE};
use super::tensor::Tensor;
use super::transpose_policy::{Contiguous, TransposePolicy};
use cblas::{dgemm, sgemm};
use road_ai_macros::expand_operations;
use typenum::{Eq, IsEqual, Unsigned};

#[expand_operations(
    dgemm<T=f64> as dot,
    sgemm<T=f32> as dot,
)]
impl<T, M, K, C, L, P> Tensor<T, Shape2D<M, K>, C, L, P>
where
    L: for<'a> Layout<'a, T>,
    C: TransposePolicy,
{
    pub fn operation<N, Crhs, Lrhs, Prhs>(
        &self,
        other: &Tensor<T, Shape2D<K, N>, Crhs, Lrhs, Prhs>,
    ) -> Tensor<T, Shape2D<M, N>, Contiguous, P::Layout, P>
    where
        M: Unsigned,
        N: Unsigned,
        K: Unsigned,
        Lrhs: for<'a> Layout<'a, T>,
        Crhs: TransposePolicy,
        P: StaticAllocationPolicy<T, Shape2D<M, N>>,
    {
        let mut out: Tensor<T, Shape2D<M, N>, Contiguous, P::Layout, P> = Tensor::default();

        unsafe {
            placeholder(
                cblas::Layout::RowMajor,
                C::BLAS_TRANSPOSE,
                Crhs::BLAS_TRANSPOSE,
                M::I32,
                N::I32,
                K::I32,
                1.0,
                self,
                K::I32,
                other,
                N::I32,
                1.0,
                &mut out,
                N::I32,
            );
        }

        out
    }

    pub fn coerce<Krhs, N, Crhs, Lrhs, Prhs>(
        &self,
        other: &Tensor<T, Shape2D<Krhs, N>, Crhs, Lrhs, Prhs>,
    ) -> Tensor<T, Shape2D<M, N>, Contiguous, P::Layout, P>
    where
        M: Unsigned,
        N: Unsigned,
        K: IsEqual<Krhs>,
        Eq<K, Krhs>: TRUE,
        Lrhs: for<'a> Layout<'a, T>,
        Crhs: TransposePolicy,
        P: StaticAllocationPolicy<T, Shape2D<M, N>>,
    {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape[1], other_shape[0],
            "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
            self_shape[1], other_shape[0], self_shape, other_shape,
        );
        let mut out: Tensor<T, Shape2D<M, N>, Contiguous, P::Layout, P> = Tensor::default();

        unsafe {
            placeholder(
                cblas::Layout::RowMajor,
                C::BLAS_TRANSPOSE,
                Crhs::BLAS_TRANSPOSE,
                M::I32,
                N::I32,
                self_shape[1] as i32,
                1.0,
                self,
                M::I32,
                other,
                N::I32,
                1.0,
                &mut out,
                M::I32,
            );
        }

        out
    }

    pub fn dynamic<Krhs, N, Crhs, Lrhs, Prhs>(
        &self,
        other: &Tensor<T, Shape2D<Krhs, N>, Crhs, Lrhs, Prhs>,
    ) -> Tensor<T, Shape2D<M, N>, Contiguous, P::Layout, P>
    where
        K: IsEqual<Krhs>,
        Eq<K, Krhs>: TRUE,
        Lrhs: for<'a> Layout<'a, T>,
        Crhs: TransposePolicy,
        P: DynamicAllocationPolicy<T>,
    {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape[1], other_shape[0],
            "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
            self_shape[1], other_shape[0], self_shape, other_shape,
        );

        let mut out: Tensor<T, Shape2D<M, N>, Contiguous, P::Layout, P> =
            Tensor::alloc(vec![self_shape[0], other_shape[1]]);

        unsafe {
            placeholder(
                cblas::Layout::RowMajor,
                C::BLAS_TRANSPOSE,
                Crhs::BLAS_TRANSPOSE,
                self_shape[0] as i32,
                other_shape[1] as i32,
                self_shape[1] as i32,
                1.0,
                self,
                self_shape[0] as i32,
                other,
                other_shape[1] as i32,
                1.0,
                &mut out,
                self_shape[0] as i32,
            );
        }

        out
    }
}
