//! `linear_algebra` contains algebra-specific operations.
//! It is currently limited to vector/vector, matrix/vector,
//! and matrix/matrix dot products. It is entirely backed by
//! openblas through C bindings.
//!
//! To avoid code duplication, this module relies on the
//! `expand_operations` procedural macro from the `melange_macro` crate.
//!
//! Note that only 1 dimmensional tensors are considered vectors
//! and that only two dimmensional tensors are considered matrices.

extern crate cblas;
extern crate openblas_src;

use super::allocation_policy::{DynamicAllocationPolicy, StaticAllocationPolicy};
use super::layout::Layout;
use super::shape::{Shape1D, Shape2D, TRUE, Static, Dynamic};
use super::tensor::Tensor;
use super::transpose_policy::{BLASPolicy, Contiguous};
use cblas::{ddot, dgemm, dgemv, sdot, sgemm, sgemv};
use melange_macros::{expand_impl, expand_trait};
use typenum::{Eq, IsEqual, Unsigned};

#[expand_trait(
    Dot, DotDynamic,
)]
pub trait Operation<Rhs> {
    type Output;

    fn operation(&self, rhs: &Rhs) -> Self::Output;
}

#[expand_impl(
    dgemm<T=f64> as dot in Dot,
    sgemm<T=f32> as dot in Dot,
)]
impl<T, M, K, C, L, P, N, Crhs, Lrhs, Prhs> Operation<Tensor<T, Shape2D<K, N>, Static, Crhs, Lrhs, Prhs>> for Tensor<T, Shape2D<M, K>, Static, C, L, P>
where
    M: Unsigned,
    N: Unsigned,
    K: Unsigned,
    C: BLASPolicy,
    L: for<'a> Layout<'a, T>,
    Lrhs: for<'a> Layout<'a, T>,
    Crhs: BLASPolicy,
    P: StaticAllocationPolicy<T, Shape2D<M, N>>,
{
    type Output = Tensor<T, Shape2D<M, N>, Static, Contiguous, P::Layout, P>;

    fn operation(
        &self,
        other: &Tensor<T, Shape2D<K, N>, Static, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        let mut out: Self::Output = Tensor::default();

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
}

#[expand_impl(
    dgemm<T=f64, D=Static, Drhs=Dynamic> as dot in Dot,
    dgemm<T=f64, D=Dynamic, Drhs=Static> as dot in Dot,
    dgemm<T=f64, D=Dynamic, Drhs=Dynamic> as dot in Dot,
    sgemm<T=f32, D=Static, Drhs=Dynamic> as dot in Dot,
    sgemm<T=f32, D=Dynamic, Drhs=Static> as dot in Dot,
    sgemm<T=f32, D=Dynamic, Drhs=Dynamic> as dot in Dot,
)]
impl<T, M, K, D, C, L, P, Krhs, N, Drhs, Crhs, Lrhs, Prhs> Operation<Tensor<T, Shape2D<Krhs, N>, Drhs, Crhs, Lrhs, Prhs>> for Tensor<T, Shape2D<M, K>, D, C, L, P>
where
    M: Unsigned,
    N: Unsigned,
    K: IsEqual<Krhs>,
    Eq<K, Krhs>: TRUE,
    C: BLASPolicy,
    L: for<'a> Layout<'a, T>,
    Lrhs: for<'a> Layout<'a, T>,
    Crhs: BLASPolicy,
    P: StaticAllocationPolicy<T, Shape2D<M, N>>,
{
    type Output = Tensor<T, Shape2D<M, N>, Static, Contiguous, P::Layout, P>;

    fn operation(
        &self,
        other: &Tensor<T, Shape2D<Krhs, N>, Drhs, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape[1], other_shape[0],
            "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
            self_shape[1], other_shape[0], self_shape, other_shape,
        );
        let mut out: Self::Output = Tensor::default();

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
                self_shape[1] as i32,
                other,
                N::I32,
                1.0,
                &mut out,
                N::I32,
            );
        }

        out
    }    
}

#[expand_impl(
    dgemm<T=f64> as dot in DotDynamic,
    sgemm<T=f32> as dot in DotDynamic,
)]
impl<T, M, K, D, C, L, P, Krhs, N, Drhs, Crhs, Lrhs, Prhs> Operation<Tensor<T, Shape2D<Krhs, N>, D, Crhs, Lrhs, Prhs>> for Tensor<T, Shape2D<M, K>, Drhs, C, L, P>
where
    K: IsEqual<Krhs>,
    Eq<K, Krhs>: TRUE,
    C: BLASPolicy,
    L: for<'a> Layout<'a, T>,
    Lrhs: for<'a> Layout<'a, T>,
    Crhs: BLASPolicy,
    P: DynamicAllocationPolicy<T>,
{
    type Output = Tensor<T, Shape2D<M, N>, Dynamic, Contiguous, P::Layout, P>;

    fn operation_dynamic(
        &self,
        other: &Tensor<T, Shape2D<Krhs, N>, D, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape[1], other_shape[0],
            "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
            self_shape[1], other_shape[0], self_shape, other_shape,
        );

        let mut out: Self::Output =
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
                self_shape[1] as i32,
                other,
                other_shape[1] as i32,
                1.0,
                &mut out,
                other_shape[1] as i32,
            );
        }

        out
    }
}

#[expand_impl(
    dgemv<T=f64> as dot in Dot,
    sgemv<T=f32> as dot in Dot,
)]
impl<T, M, N, C, L, P, Crhs, Lrhs, Prhs> Operation<Tensor<T, Shape1D<N>, Static, Crhs, Lrhs, Prhs>> for Tensor<T, Shape2D<M, N>, Static, C, L, P>
where
    M: Unsigned,
    N: Unsigned,    
    L: for<'a> Layout<'a, T>,
    C: BLASPolicy,
    Lrhs: for<'a> Layout<'a, T>,
    P: StaticAllocationPolicy<T, Shape1D<M>>,
{
    type Output = Tensor<T, Shape1D<M>, Static, Contiguous, P::Layout, P>;

    fn operation(
        &self,
        other: &Tensor<T, Shape1D<N>, Static, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        let mut out: Self::Output = Tensor::default();

        unsafe {
            placeholder(
                cblas::Layout::RowMajor,
                C::BLAS_TRANSPOSE,
                M::I32,
                N::I32,
                1.0,
                self,
                N::I32,
                other,
                1,
                1.0,
                &mut out,
                1,
            );
        }

        out
    }
}

#[expand_impl(
    dgemv<T=f64, D=Static, Drhs=Dynamic> as dot in Dot,
    dgemv<T=f64, D=Dynamic, Drhs=Static> as dot in Dot,
    dgemv<T=f64, D=Dynamic, Drhs=Dynamic> as dot in Dot,
    sgemv<T=f32, D=Static, Drhs=Dynamic> as dot in Dot,
    sgemv<T=f32, D=Dynamic, Drhs=Static> as dot in Dot,
    sgemv<T=f32, D=Dynamic, Drhs=Dynamic> as dot in Dot,
)]
impl<T, M, N, D, C, L, P, Nrhs, Drhs, Crhs, Lrhs, Prhs> Operation<Tensor<T, Shape1D<Nrhs>, Drhs, Crhs, Lrhs, Prhs>> for Tensor<T, Shape2D<M, N>, D, C, L, P>
where
    M: Unsigned,
    N: IsEqual<Nrhs>,
    Eq<N, Nrhs>: TRUE,    
    L: for<'a> Layout<'a, T>,
    C: BLASPolicy,
    Lrhs: for<'a> Layout<'a, T>,
    P: StaticAllocationPolicy<T, Shape1D<M>>,
{
    type Output = Tensor<T, Shape1D<M>, Static, Contiguous, P::Layout, P>;

    fn operation(
        &self,
        other: &Tensor<T, Shape1D<Nrhs>, Drhs, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape[1], other_shape[0],
            "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
            self_shape[1], other_shape[0], self_shape, other_shape,
        );
        let mut out: Self::Output = Tensor::default();

        unsafe {
            placeholder(
                cblas::Layout::RowMajor,
                C::BLAS_TRANSPOSE,
                M::I32,
                self_shape[1] as i32,
                1.0,
                self,
                self_shape[1] as i32,
                other,
                1,
                1.0,
                &mut out,
                1,
            );
        }

        out
    }
}

#[expand_impl(
    dgemv<T=f64> as dot in DotDynamic,
    sgemv<T=f32> as dot in DotDynamic,
)]
impl<T, M, N, D, C, L, P, Nrhs, Drhs, Crhs, Lrhs, Prhs> Operation<Tensor<T, Shape1D<Nrhs>, Drhs, Crhs, Lrhs, Prhs>> for Tensor<T, Shape2D<M, N>, D, C, L, P>
where
    N: IsEqual<Nrhs>,
    Eq<N, Nrhs>: TRUE,    
    L: for<'a> Layout<'a, T>,
    C: BLASPolicy,
    Lrhs: for<'a> Layout<'a, T>,
    P: DynamicAllocationPolicy<T>,
{
    type Output = Tensor<T, Shape1D<M>, Dynamic, Contiguous, P::Layout, P>;

    fn operation_dynamic(
        &self,
        other: &Tensor<T, Shape1D<Nrhs>, Drhs, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape[1], other_shape[0],
            "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
            self_shape[1], other_shape[0], self_shape, other_shape,
        );

        let mut out: Self::Output =
            Tensor::alloc(vec![self_shape[0]]);

        unsafe {
            placeholder(
                cblas::Layout::RowMajor,
                C::BLAS_TRANSPOSE,
                self_shape[0] as i32,
                self_shape[1] as i32,
                1.0,
                self,
                self_shape[1] as i32,
                other,
                1,
                1.0,
                &mut out,
                1,
            );
        }

        out
    }
}

#[expand_impl(
    ddot<T=f64> as dot in Dot,
    sdot<T=f32> as dot in Dot,
)]
impl<T, N, C, L, P, Crhs, Lrhs, Prhs> Operation<Tensor<T, Shape1D<N>, Static, Crhs, Lrhs, Prhs>> for Tensor<T, Shape1D<N>, Static, C, L, P>
where
    N: Unsigned,
    L: for<'a> Layout<'a, T>,
    C: BLASPolicy,
    Lrhs: for<'a> Layout<'a, T>,
{
    type Output = T;

    fn operation(&self, other: &Tensor<T, Shape1D<N>, Static, Crhs, Lrhs, Prhs>) -> T {
        unsafe { placeholder(N::I32, self, 1, other, 1) }
    }
}

#[expand_impl(
    ddot<T=f64, D=Static, Drhs=Dynamic> as dot in Dot,
    ddot<T=f64, D=Dynamic, Drhs=Static> as dot in Dot,
    ddot<T=f64, D=Dynamic, Drhs=Dynamic> as dot in Dot,
    sdot<T=f32, D=Static, Drhs=Dynamic> as dot in Dot,
    sdot<T=f32, D=Dynamic, Drhs=Static> as dot in Dot,
    sdot<T=f32, D=Dynamic, Drhs=Dynamic> as dot in Dot,
)]
impl<T, N, D, C, L, P, Drhs, Nrhs, Crhs, Lrhs, Prhs> Operation<Tensor<T, Shape1D<Nrhs>, Drhs, Crhs, Lrhs, Prhs>> for Tensor<T, Shape1D<N>, D, C, L, P>
where
    N: IsEqual<Nrhs>,
    Eq<N, Nrhs>: TRUE,
    L: for<'a> Layout<'a, T>,
    C: BLASPolicy,
    Lrhs: for<'a> Layout<'a, T>,
{
    type Output = T;
    
    fn operation(&self, other: &Tensor<T, Shape1D<Nrhs>, Drhs, Crhs, Lrhs, Prhs>) -> T {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape[0], other_shape[0],
            "Contracted dimmensions {} and {} must be equal, got shapes {:?} and {:?}.",
            self_shape[0], other_shape[0], self_shape, other_shape,
        );

        unsafe { placeholder(self_shape[0] as i32, self, 1, other, 1) }
    }
}
