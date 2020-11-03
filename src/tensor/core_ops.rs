//! `core_ops` contains basic mathematical operations at the tensor level
//! that rely on the implementation of the underlying data type `T` of the
//! tensor. There are two families of operations: functionnal operations
//! and in-place operations.
//! 
//! Functionnal operations are methods that immutably borrow `self` and
//! optionnaly borrow other tensors. They return new tensors that have been
//! allocated following the allocation policy of `self` tensor.
//! Conversely, in-place operations mutably borrow `self` and optionnaly
//! immutably borrow other tensors to mutate directly mutate `self` data.
//! 
//! Functionnal operations are interesting when backpropagating because they
//! preserve operands whereas in-place operations reduce the memory footprint.
//! 
//! Both families of operations perform ad-hoc parallel computation
//! acording to how data is stored. Note that operations on more than one
//! tensors require all tensors to have compatible shapes (all dimensions
//! must be equal or `Dyn`). If this is not the case consider broadcasting.
//!
//! Due to the large amount of duplicate code
//! between all the operations, this module relies on the `expand_operations`
//! procedural macros defined in the `melange_macros` crate. This macro is
//! able to adapt generic operation code (such as loops and iterators) to
//! specific operations requirering specific bounds on `T` or its
//! replacement with a concrete type like `f64`.
//!
//! Note that some methods are only available for tensors based on float
//! types `f32` and `f64`. This is inherent to how numeric types are treated
//! in rust.
//! 
//! Please refer to the definition of the scalar version of the mathematical
//! operation in `std` for more.

use super::allocation_policy::{DynamicAllocationPolicy, StaticAllocationPolicy};
use super::layout::{Layout, LayoutMut};
use super::shape::{ReprShape, ReprShapeDyn, Same, TRUE, Static, Dynamic};
use super::tensor::Tensor;
use super::transpose_policy::Contiguous;
use crate::ring::Ring;
use rayon::prelude::*;
use melange_macros::{expand_impl, expand_trait};
use std::ops::*;

fn assert_shape_eq(lhs: Vec<usize>, rhs: Vec<usize>) {
    assert_eq!(
        lhs,
        rhs,
        "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
        lhs,
        rhs
    );
}

#[expand_trait(
    Atan2, Copysign, DivEuclid, Max, Min, RemEuclid,
    AddDynamic, SubDynamic, MulDynamic, DivDynamic, RemDynamic, Atan2Dynamic,
    CopysignDynamic, DivEuclidDynamic, MaxDynamic, MinDynamic, RemEuclidDynamic,
)]
pub trait Operation<Rhs> {
    type Output;

    fn operation(self, rhs: Rhs) -> Self::Output;
}

#[expand_trait(
    Add_, Sub_, Mul_, Div_, Rem_, Atan2_, Copysign_, DivEuclid_, Max_, Min_, RemEuclid_,
)]
pub trait Operation<Rhs> {
    fn operation(&mut self, rhs: Rhs);
}

#[expand_trait(MulAdd, MulAddDynamic)]
pub trait Operation<Rhs0, Rhs1> {
    type Output;

    fn operation(self, rhs0: Rhs0, rhs1: Rhs1) -> Self::Output;
}

#[expand_trait(MulAdd_, MulAddDynamic_)]
pub trait Operation<Rhs0, Rhs1> {
    fn operation(&mut self, rhs0: Rhs0, rhs1: Rhs1);
}

#[expand_impl(
    add<T: Send + Sync + Copy + Add<Output=T>>,
    sub<T: Send + Sync + Copy + Sub<Output=T>>,
    mul<T: Send + Sync + Copy + Mul<Output=T>>,
    div<T: Send + Sync + Copy + Div<Output=T>>,
    rem<T: Send + Sync + Copy + Rem<Output=T>>,
    atan2<T=f64>,
    copysign<T=f64>,
    div_euclid<T=f64>,
    max<T=f64>,
    min<T=f64>,
    rem_euclid<T=f64>,
    atan2<T=f32>,
    copysign<T=f32>,
    div_euclid<T=f32>,
    max<T=f32>,
    min<T=f32>,
    rem_euclid<T=f32>,
    div_euclid<T=u128>,
    rem_euclid<T=u128>,
    div_euclid<T=u64>,
    rem_euclid<T=u64>,
    div_euclid<T=u32>,
    rem_euclid<T=u32>,
    div_euclid<T=u16>,
    rem_euclid<T=u16>,
    div_euclid<T=u8>,
    rem_euclid<T=u8>,
    div_euclid<T=i128>,
    rem_euclid<T=i128>,
    div_euclid<T=i64>,
    rem_euclid<T=i64>,
    div_euclid<T=i32>,
    rem_euclid<T=i32>,
    div_euclid<T=i16>,
    rem_euclid<T=i16>,
    div_euclid<T=i8>,
    rem_euclid<T=i8>,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked<Srhs, Drhs, Crhs, Lrhs, Prhs, Sout, Dout, Lout>(
        &self,
        rhs: &Tensor<T, Srhs, Drhs, Crhs, Lrhs, Prhs>,
        out: &mut Tensor<T, Sout, Dout, Contiguous, Lout, P>,
    ) where
        L: for<'a> Layout<'a, T>,
        Lrhs: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size().min(rhs.opt_chunk_size());

        for ((chunk_self, chunck_rhs), chunk_out) in self
            .chunks(chunk_size)
            .zip(rhs.chunks(chunk_size))
            .zip(out.chunks_mut(chunk_size))
        {
            chunk_self
                .par_iter()
                .zip(chunck_rhs.par_iter())
                .zip(chunk_out.par_iter_mut())
                .for_each(|((x, y), z)| *z = x.placeholder(*y));
        }
    }
}

#[expand_impl(
    add<T: Send + Sync + Copy + Add<Output=T>> in Add,
    sub<T: Send + Sync + Copy + Sub<Output=T>> in Sub,
    mul<T: Send + Sync + Copy + Mul<Output=T>> in Mul,
    div<T: Send + Sync + Copy + Div<Output=T>> in Div,
    rem<T: Send + Sync + Copy + Rem<Output=T>> in Rem,
    atan2<T=f64> in Atan2,
    copysign<T=f64> in Copysign,
    div_euclid<T=f64> in DivEuclid,
    max<T=f64> in Max,
    min<T=f64> in Min,
    rem_euclid<T=f64> in RemEuclid,
    atan2<T=f32> in Atan2,
    copysign<T=f32> in Copysign,
    div_euclid<T=f32> in DivEuclid,
    max<T=f32> in Max,
    min<T=f32> in Min,
    rem_euclid<T=f32> in RemEuclid,
    div_euclid<T=u128> in DivEuclid,
    rem_euclid<T=u128> in RemEuclid,
    div_euclid<T=u64> in DivEuclid,
    rem_euclid<T=u64> in RemEuclid,
    div_euclid<T=u32> in DivEuclid,
    rem_euclid<T=u32> in RemEuclid,
    div_euclid<T=u16> in DivEuclid,
    rem_euclid<T=u16> in RemEuclid,
    div_euclid<T=u8> in DivEuclid,
    rem_euclid<T=u8> in RemEuclid,
    div_euclid<T=i128> in DivEuclid,
    rem_euclid<T=i128> in RemEuclid,
    div_euclid<T=i64> in DivEuclid,
    rem_euclid<T=i64> in RemEuclid,
    div_euclid<T=i32> in DivEuclid,
    rem_euclid<T=i32> in RemEuclid,
    div_euclid<T=i16> in DivEuclid,
    rem_euclid<T=i16> in RemEuclid,
    div_euclid<T=i8> in DivEuclid,
    rem_euclid<T=i8> in RemEuclid,
)]
impl<T, S, C, L, P, Crhs, Lrhs, Prhs> Operation<&Tensor<T, S, Static, Crhs, Lrhs, Prhs>> for &Tensor<T, S, Static, C, L, P>
where
    L: for<'a> Layout<'a, T>,
    P: StaticAllocationPolicy<T, S>,
    Lrhs: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, S, Static, Contiguous, P::Layout, P>;
    
    fn operation(
        self,
        rhs: &Tensor<T, S, Static, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        let mut out = Tensor::default();
        self.unchecked(rhs, &mut out);

        out
    }
}

#[expand_impl(
    add<T: Send + Sync + Copy + Add<Output=T>> in Add,
    sub<T: Send + Sync + Copy + Sub<Output=T>> in Sub,
    mul<T: Send + Sync + Copy + Mul<Output=T>> in Mul,
    div<T: Send + Sync + Copy + Div<Output=T>> in Div,
    rem<T: Send + Sync + Copy + Rem<Output=T>> in Rem,
    atan2<T=f64> in Atan2,
    copysign<T=f64> in Copysign,
    div_euclid<T=f64> in DivEuclid,
    max<T=f64> in Max,
    min<T=f64> in Min,
    rem_euclid<T=f64> in RemEuclid,
    atan2<T=f32> in Atan2,
    copysign<T=f32> in Copysign,
    div_euclid<T=f32> in DivEuclid,
    max<T=f32> in Max,
    min<T=f32> in Min,
    rem_euclid<T=f32> in RemEuclid,
    div_euclid<T=u128> in DivEuclid,
    rem_euclid<T=u128> in RemEuclid,
    div_euclid<T=u64> in DivEuclid,
    rem_euclid<T=u64> in RemEuclid,
    div_euclid<T=u32> in DivEuclid,
    rem_euclid<T=u32> in RemEuclid,
    div_euclid<T=u16> in DivEuclid,
    rem_euclid<T=u16> in RemEuclid,
    div_euclid<T=u8> in DivEuclid,
    rem_euclid<T=u8> in RemEuclid,
    div_euclid<T=i128> in DivEuclid,
    rem_euclid<T=i128> in RemEuclid,
    div_euclid<T=i64> in DivEuclid,
    rem_euclid<T=i64> in RemEuclid,
    div_euclid<T=i32> in DivEuclid,
    rem_euclid<T=i32> in RemEuclid,
    div_euclid<T=i16> in DivEuclid,
    rem_euclid<T=i16> in RemEuclid,
    div_euclid<T=i8> in DivEuclid,
    rem_euclid<T=i8> in RemEuclid,
)]
#[expand_impl(
    _a<D=Static, Drhs=Dynamic, Sout=S>,
    _b<D=Dynamic, Drhs=Static, Sout=Srhs>,
    _c<D=Dynamic, Drhs=Dynamic, S: ReprShape<T, Srhs>, Sout=<S as ReprShape<T, Srhs>>::Output>
)]
impl<T, S, C, L, P, Srhs, Crhs, Lrhs, Prhs> Operation<&Tensor<T, Srhs, Drhs, Crhs, Lrhs, Prhs>> for &Tensor<T, S, D, C, L, P>
where
    S: Same<Srhs>,
    <S as Same<Srhs>>::Output: TRUE,
    L: for<'a> Layout<'a, T>,
    P: StaticAllocationPolicy<T, Sout>,
    Lrhs: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, Sout, Static, Contiguous, P::Layout, P>;

    fn operation(
        self,
        rhs: &Tensor<T, Srhs, Drhs, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        assert_shape_eq(self.shape(), rhs.shape());

        let mut out = Tensor::default();
        self.unchecked(rhs, &mut out);

        out
    }
}

#[expand_impl(
    add<T: Send + Sync + Copy + Add<Output=T>> in AddDynamic,
    sub<T: Send + Sync + Copy + Sub<Output=T>> in SubDynamic,
    mul<T: Send + Sync + Copy + Mul<Output=T>> in MulDynamic,
    div<T: Send + Sync + Copy + Div<Output=T>> in DivDynamic,
    rem<T: Send + Sync + Copy + Rem<Output=T>> in RemDynamic,
    atan2<T=f64> in Atan2Dynamic,
    copysign<T=f64> in CopysignDynamic,
    div_euclid<T=f64> in DivEuclidDynamic,
    max<T=f64> in MaxDynamic,
    min<T=f64> in MinDynamic,
    rem_euclid<T=f64> in RemEuclidDynamic,
    atan2<T=f32> in Atan2Dynamic,
    copysign<T=f32> in CopysignDynamic,
    div_euclid<T=f32> in DivEuclidDynamic,
    max<T=f32> in MaxDynamic,
    min<T=f32> in MinDynamic,
    rem_euclid<T=f32> in RemEuclidDynamic,
    div_euclid<T=u128> in DivEuclidDynamic,
    rem_euclid<T=u128> in RemEuclidDynamic,
    div_euclid<T=u64> in DivEuclidDynamic,
    rem_euclid<T=u64> in RemEuclidDynamic,
    div_euclid<T=u32> in DivEuclidDynamic,
    rem_euclid<T=u32> in RemEuclidDynamic,
    div_euclid<T=u16> in DivEuclidDynamic,
    rem_euclid<T=u16> in RemEuclidDynamic,
    div_euclid<T=u8> in DivEuclidDynamic,
    rem_euclid<T=u8> in RemEuclidDynamic,
    div_euclid<T=i128> in DivEuclidDynamic,
    rem_euclid<T=i128> in RemEuclidDynamic,
    div_euclid<T=i64> in DivEuclidDynamic,
    rem_euclid<T=i64> in RemEuclidDynamic,
    div_euclid<T=i32> in DivEuclidDynamic,
    rem_euclid<T=i32> in RemEuclidDynamic,
    div_euclid<T=i16> in DivEuclidDynamic,
    rem_euclid<T=i16> in RemEuclidDynamic,
    div_euclid<T=i8> in DivEuclidDynamic,
    rem_euclid<T=i8> in RemEuclidDynamic,
)]
impl<T, S, C, L, P, Srhs, Crhs, Lrhs, Prhs> Operation<&Tensor<T, Srhs, Dynamic, Crhs, Lrhs, Prhs>> for &Tensor<T, S, Dynamic, C, L, P>
where
    S: Same<Srhs> + ReprShapeDyn<T, Srhs>,
    <S as Same<Srhs>>::Output: TRUE,
    L: for<'a> Layout<'a, T>,
    P: DynamicAllocationPolicy<T>,
    Lrhs: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, <S as ReprShapeDyn<T, Srhs>>::Output, Dynamic, Contiguous, P::Layout, P>;

    fn operation_dynamic(
        self,
        rhs: &Tensor<T, Srhs, Dynamic, Crhs, Lrhs, Prhs>,
    ) -> Self::Output {
        assert_shape_eq(self.shape(), rhs.shape());

        let mut out = Tensor::alloc(self.shape());
        self.unchecked(rhs, &mut out);

        out
    }
}

#[expand_impl(
    add<T: Send + Sync + Copy + Add<Output=T>>(T) as scal_add,
    sub<T: Send + Sync + Copy + Sub<Output=T>>(T) as scal_sub,
    mul<T: Send + Sync + Copy + Mul<Output=T>>(T) as scal_mul,
    div<T: Send + Sync + Copy + Div<Output=T>>(T) as scal_div,
    rem<T: Send + Sync + Copy + Rem<Output=T>>(T) as scal_rem,
    div_euclid<T=f64>(f64) as scal_div_euclid,
    max<T=f64>(f64) as scal_max,
    min<T=f64>(f64) as scal_min,
    powf<T=f64>(f64),
    rem_euclid<T=f64>(f64) as scal_rem_euclid,
    powi<T=f64>(i32),
    max<T=f32>(f32) as scal_max,
    min<T=f32>(f32) as scal_min,
    powf<T=f32>(f32),
    rem_euclid<T=f32>(f32) as scal_rem_euclid,
    powi<T=f32>(i32),
    div_euclid<T=u128>(u128) as scal_div_euclid,
    rem_euclid<T=u128>(u128) as scal_rem_euclid,
    div_euclid<T=u64>(u64) as scal_div_euclid,
    rem_euclid<T=u64>(u64) as scal_rem_euclid,
    div_euclid<T=u32>(u32) as scal_div_euclid,
    rem_euclid<T=u32>(u32) as scal_rem_euclid,
    div_euclid<T=u16>(u16) as scal_div_euclid,
    rem_euclid<T=u16>(u16) as scal_rem_euclid,
    div_euclid<T=u8>(u8) as scal_div_euclid,
    rem_euclid<T=u8>(u8) as scal_rem_euclid,
    div_euclid<T=i128>(i128) as scal_div_euclid,
    rem_euclid<T=i128>(i128) as scal_rem_euclid,
    div_euclid<T=i64>(i64) as scal_div_euclid,
    rem_euclid<T=i64>(i64) as scal_rem_euclid,
    div_euclid<T=i32>(i32) as scal_div_euclid,
    rem_euclid<T=i32>(i32) as scal_rem_euclid,
    div_euclid<T=i16>(i16) as scal_div_euclid,
    rem_euclid<T=i16>(i16) as scal_rem_euclid,
    div_euclid<T=i8>(i8) as scal_div_euclid,
    rem_euclid<T=i8>(i8) as scal_rem_euclid,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked<Lout>(&self, param: type0, out: &mut Tensor<T, S, D, Contiguous, Lout, P>)
    where
        L: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();

        for (chunk_self, chunk_out) in self.chunks(chunk_size).zip(out.chunks_mut(chunk_size)) {
            chunk_self
                .par_iter()
                .zip(chunk_out.par_iter_mut())
                .for_each(|(x, y)| *y = x.placeholder(param));
        }
    }
}

#[expand_impl(
    add<T: Send + Sync + Copy + Add<Output=T>>(T) as scal_add,
    sub<T: Send + Sync + Copy + Sub<Output=T>>(T) as scal_sub,
    mul<T: Send + Sync + Copy + Mul<Output=T>>(T) as scal_mul,
    div<T: Send + Sync + Copy + Div<Output=T>>(T) as scal_div,
    rem<T: Send + Sync + Copy + Rem<Output=T>>(T) as scal_rem,
    div_euclid<T=f64>(f64) as scal_div_euclid,
    max<T=f64>(f64) as scal_max,
    min<T=f64>(f64) as scal_min,
    powf<T=f64>(f64),
    rem_euclid<T=f64>(f64) as scal_rem_euclid,
    powi<T=f64>(i32),
    max<T=f32>(f32) as scal_max,
    min<T=f32>(f32) as scal_min,
    powf<T=f32>(f32),
    rem_euclid<T=f32>(f32) as scal_rem_euclid,
    powi<T=f32>(i32),
    div_euclid<T=u128>(u128) as scal_div_euclid,
    rem_euclid<T=u128>(u128) as scal_rem_euclid,
    div_euclid<T=u64>(u64) as scal_div_euclid,
    rem_euclid<T=u64>(u64) as scal_rem_euclid,
    div_euclid<T=u32>(u32) as scal_div_euclid,
    rem_euclid<T=u32>(u32) as scal_rem_euclid,
    div_euclid<T=u16>(u16) as scal_div_euclid,
    rem_euclid<T=u16>(u16) as scal_rem_euclid,
    div_euclid<T=u8>(u8) as scal_div_euclid,
    rem_euclid<T=u8>(u8) as scal_rem_euclid,
    div_euclid<T=i128>(i128) as scal_div_euclid,
    rem_euclid<T=i128>(i128) as scal_rem_euclid,
    div_euclid<T=i64>(i64) as scal_div_euclid,
    rem_euclid<T=i64>(i64) as scal_rem_euclid,
    div_euclid<T=i32>(i32) as scal_div_euclid,
    rem_euclid<T=i32>(i32) as scal_rem_euclid,
    div_euclid<T=i16>(i16) as scal_div_euclid,
    rem_euclid<T=i16>(i16) as scal_rem_euclid,
    div_euclid<T=i8>(i8) as scal_div_euclid,
    rem_euclid<T=i8>(i8) as scal_rem_euclid,
)]
impl<T, S, C, L, P> Tensor<T, S, Static, C, L, P> {
    pub fn operation(&self, param: type0) -> Tensor<T, S, Static, Contiguous, P::Layout, P>
    where
        L: for<'a> Layout<'a, T>,
        P: StaticAllocationPolicy<T, S>,
    {
        let mut out = Tensor::default();

        self.unchecked(param, &mut out);

        out
    }
}

#[expand_impl(
    add<T: Send + Sync + Copy + Add<Output=T>>(T) as scal_add,
    sub<T: Send + Sync + Copy + Sub<Output=T>>(T) as scal_sub,
    mul<T: Send + Sync + Copy + Mul<Output=T>>(T) as scal_mul,
    div<T: Send + Sync + Copy + Div<Output=T>>(T) as scal_div,
    rem<T: Send + Sync + Copy + Rem<Output=T>>(T) as scal_rem,
    div_euclid<T=f64>(f64) as scal_div_euclid,
    max<T=f64>(f64) as scal_max,
    min<T=f64>(f64) as scal_min,
    powf<T=f64>(f64),
    rem_euclid<T=f64>(f64) as scal_rem_euclid,
    powi<T=f64>(i32),
    max<T=f32>(f32) as scal_max,
    min<T=f32>(f32) as scal_min,
    powf<T=f32>(f32),
    rem_euclid<T=f32>(f32) as scal_rem_euclid,
    powi<T=f32>(i32),
    div_euclid<T=u128>(u128) as scal_div_euclid,
    rem_euclid<T=u128>(u128) as scal_rem_euclid,
    div_euclid<T=u64>(u64) as scal_div_euclid,
    rem_euclid<T=u64>(u64) as scal_rem_euclid,
    div_euclid<T=u32>(u32) as scal_div_euclid,
    rem_euclid<T=u32>(u32) as scal_rem_euclid,
    div_euclid<T=u16>(u16) as scal_div_euclid,
    rem_euclid<T=u16>(u16) as scal_rem_euclid,
    div_euclid<T=u8>(u8) as scal_div_euclid,
    rem_euclid<T=u8>(u8) as scal_rem_euclid,
    div_euclid<T=i128>(i128) as scal_div_euclid,
    rem_euclid<T=i128>(i128) as scal_rem_euclid,
    div_euclid<T=i64>(i64) as scal_div_euclid,
    rem_euclid<T=i64>(i64) as scal_rem_euclid,
    div_euclid<T=i32>(i32) as scal_div_euclid,
    rem_euclid<T=i32>(i32) as scal_rem_euclid,
    div_euclid<T=i16>(i16) as scal_div_euclid,
    rem_euclid<T=i16>(i16) as scal_rem_euclid,
    div_euclid<T=i8>(i8) as scal_div_euclid,
    rem_euclid<T=i8>(i8) as scal_rem_euclid,
)]
impl<T, S, C, L, P> Tensor<T, S, Dynamic, C, L, P> {
    pub fn operation(&self, param: type0) -> Tensor<T, S, Dynamic, Contiguous, P::Layout, P>
    where
        L: for<'a> Layout<'a, T>,
        P: DynamicAllocationPolicy<T>,
    {
        let mut out = Tensor::alloc(self.shape());

        self.unchecked(param, &mut out);

        out
    }
}

#[expand_impl(
    exp<T=f64>,
    exp2<T=f64>,
    exp_m1<T=f64>,
    ln<T=f64>,
    ln_1p<T=f64>,
    log2<T=f64>,
    log10<T=f64>,
    sin<T=f64>,
    cos<T=f64>,
    tan<T=f64>,
    sinh<T=f64>,
    cosh<T=f64>,
    tanh<T=f64>,
    asin<T=f64>,
    acos<T=f64>,
    atan<T=f64>,
    asinh<T=f64>,
    acosh<T=f64>,
    atanh<T=f64>,
    sqrt<T=f64>,
    cbrt<T=f64>,
    abs<T=f64>,
    signum<T=f64>,
    ceil<T=f64>,
    floor<T=f64>,
    round<T=f64>,
    recip<T=f64>,
    to_degrees<T=f64>,
    to_radians<T=f64>,
    exp<T=f32>,
    exp2<T=f32>,
    exp_m1<T=f32>,
    ln<T=f32>,
    ln_1p<T=f32>,
    log2<T=f32>,
    log10<T=f32>,
    sin<T=f32>,
    cos<T=f32>,
    tan<T=f32>,
    sinh<T=f32>,
    cosh<T=f32>,
    tanh<T=f32>,
    asin<T=f32>,
    acos<T=f32>,
    atan<T=f32>,
    asinh<T=f32>,
    acosh<T=f32>,
    atanh<T=f32>,
    sqrt<T=f32>,
    cbrt<T=f32>,
    abs<T=f32>,
    signum<T=f32>,
    ceil<T=f32>,
    floor<T=f32>,
    round<T=f32>,
    recip<T=f32>,
    to_degrees<T=f32>,
    to_radians<T=f32>,
    abs<T=i128>,
    signum<T=i128>,
    abs<T=i64>,
    signum<T=i64>,
    abs<T=i32>,
    signum<T=i32>,
    abs<T=i16>,
    signum<T=i16>,
    abs<T=i8>,
    signum<T=i8>,
    clone<T: Send + Sync + Copy> as as_contiguous,
    inv<T: Send + Sync + Copy + Ring + Div<Output = T>>,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked<Lout>(&self, out: &mut Tensor<T, S, D, Contiguous, Lout, P>)
    where
        L: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();

        for (chunk_self, chunk_out) in self.chunks(chunk_size).zip(out.chunks_mut(chunk_size)) {
            chunk_self
                .par_iter()
                .zip(chunk_out.par_iter_mut())
                .for_each(|(x, y)| *y = x.placeholder());
        }
    }
}

#[expand_impl(
    exp<T=f64>,
    exp2<T=f64>,
    exp_m1<T=f64>,
    ln<T=f64>,
    ln_1p<T=f64>,
    log2<T=f64>,
    log10<T=f64>,
    sin<T=f64>,
    cos<T=f64>,
    tan<T=f64>,
    sinh<T=f64>,
    cosh<T=f64>,
    tanh<T=f64>,
    asin<T=f64>,
    acos<T=f64>,
    atan<T=f64>,
    asinh<T=f64>,
    acosh<T=f64>,
    atanh<T=f64>,
    sqrt<T=f64>,
    cbrt<T=f64>,
    abs<T=f64>,
    signum<T=f64>,
    ceil<T=f64>,
    floor<T=f64>,
    round<T=f64>,
    recip<T=f64>,
    to_degrees<T=f64>,
    to_radians<T=f64>,
    exp<T=f32>,
    exp2<T=f32>,
    exp_m1<T=f32>,
    ln<T=f32>,
    ln_1p<T=f32>,
    log2<T=f32>,
    log10<T=f32>,
    sin<T=f32>,
    cos<T=f32>,
    tan<T=f32>,
    sinh<T=f32>,
    cosh<T=f32>,
    tanh<T=f32>,
    asin<T=f32>,
    acos<T=f32>,
    atan<T=f32>,
    asinh<T=f32>,
    acosh<T=f32>,
    atanh<T=f32>,
    sqrt<T=f32>,
    cbrt<T=f32>,
    abs<T=f32>,
    signum<T=f32>,
    ceil<T=f32>,
    floor<T=f32>,
    round<T=f32>,
    recip<T=f32>,
    to_degrees<T=f32>,
    to_radians<T=f32>,
    abs<T=i128>,
    signum<T=i128>,
    abs<T=i64>,
    signum<T=i64>,
    abs<T=i32>,
    signum<T=i32>,
    abs<T=i16>,
    signum<T=i16>,
    abs<T=i8>,
    signum<T=i8>,
    clone<T: Send + Sync + Copy> as as_contiguous,
    inv<T: Send + Sync + Copy + Ring + Div<Output = T>>,
)]
impl<T, S, C, L, P> Tensor<T, S, Static, C, L, P> {
    pub fn operation(&self) -> Tensor<T, S, Static, Contiguous, P::Layout, P>
    where
        L: for<'a> Layout<'a, T>,
        P: StaticAllocationPolicy<T, S>,
    {
        let mut out = Tensor::default();

        self.unchecked(&mut out);

        out
    }
}

#[expand_impl(
    exp<T=f64>,
    exp2<T=f64>,
    exp_m1<T=f64>,
    ln<T=f64>,
    ln_1p<T=f64>,
    log2<T=f64>,
    log10<T=f64>,
    sin<T=f64>,
    cos<T=f64>,
    tan<T=f64>,
    sinh<T=f64>,
    cosh<T=f64>,
    tanh<T=f64>,
    asin<T=f64>,
    acos<T=f64>,
    atan<T=f64>,
    asinh<T=f64>,
    acosh<T=f64>,
    atanh<T=f64>,
    sqrt<T=f64>,
    cbrt<T=f64>,
    abs<T=f64>,
    signum<T=f64>,
    ceil<T=f64>,
    floor<T=f64>,
    round<T=f64>,
    recip<T=f64>,
    to_degrees<T=f64>,
    to_radians<T=f64>,
    exp<T=f32>,
    exp2<T=f32>,
    exp_m1<T=f32>,
    ln<T=f32>,
    ln_1p<T=f32>,
    log2<T=f32>,
    log10<T=f32>,
    sin<T=f32>,
    cos<T=f32>,
    tan<T=f32>,
    sinh<T=f32>,
    cosh<T=f32>,
    tanh<T=f32>,
    asin<T=f32>,
    acos<T=f32>,
    atan<T=f32>,
    asinh<T=f32>,
    acosh<T=f32>,
    atanh<T=f32>,
    sqrt<T=f32>,
    cbrt<T=f32>,
    abs<T=f32>,
    signum<T=f32>,
    ceil<T=f32>,
    floor<T=f32>,
    round<T=f32>,
    recip<T=f32>,
    to_degrees<T=f32>,
    to_radians<T=f32>,
    abs<T=i128>,
    signum<T=i128>,
    abs<T=i64>,
    signum<T=i64>,
    abs<T=i32>,
    signum<T=i32>,
    abs<T=i16>,
    signum<T=i16>,
    abs<T=i8>,
    signum<T=i8>,
    clone<T: Send + Sync + Copy> as as_contiguous,
    inv<T: Send + Sync + Copy + Ring + Div<Output = T>>,
)]
impl<T, S, C, L, P> Tensor<T, S, Dynamic, C, L, P> {
    pub fn operation(&self) -> Tensor<T, S, Dynamic, Contiguous, P::Layout, P>
    where
        L: for<'a> Layout<'a, T>,
        P: DynamicAllocationPolicy<T>,
    {
        let mut out = Tensor::alloc(self.shape());

        self.unchecked(&mut out);

        out
    }
}

#[expand_impl(
    mul_add<T=f64>(f64, f64) as scal_mul_add,
    mul_add<T=f32>(f32, f32) as scal_mul_add,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked<Lout>(
        &self,
        param0: type0,
        param1: type1,
        out: &mut Tensor<T, S, D, Contiguous, Lout, P>,
    ) where
        L: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();

        for (chunk_self, chunk_out) in self.chunks(chunk_size).zip(out.chunks_mut(chunk_size)) {
            chunk_self
                .par_iter()
                .zip(chunk_out.par_iter_mut())
                .for_each(|(x, y)| *y = x.placeholder(param0, param1));
        }
    }
}

#[expand_impl(
    mul_add<T=f64>(f64, f64) as scal_mul_add,
    mul_add<T=f32>(f32, f32) as scal_mul_add,
)]
impl<T, S, C, L, P> Tensor<T, S, Static, C, L, P> {
    pub fn operation(&self, param0: type0, param1: type1) -> Tensor<T, S, Static, Contiguous, P::Layout, P>
    where
        L: for<'a> Layout<'a, T>,
        P: StaticAllocationPolicy<T, S>,
    {
        let mut out = Tensor::default();

        self.unchecked(param0, param1, &mut out);

        out
    }
}

#[expand_impl(
    mul_add<T=f64>(f64, f64) as scal_mul_add,
    mul_add<T=f32>(f32, f32) as scal_mul_add,
)]
impl<T, S, C, L, P> Tensor<T, S, Dynamic, C, L, P> {
    pub fn operation(&self, param0: type0, param1: type1) -> Tensor<T, S, Dynamic, Contiguous, P::Layout, P>
    where
        L: for<'a> Layout<'a, T>,
        P: DynamicAllocationPolicy<T>,
    {
        let mut out = Tensor::alloc(self.shape());

        self.unchecked(param0, param1, &mut out);

        out
    }
}

#[expand_impl(
    mul_add<T=f64>,
    mul_add<T=f32>,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked<Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1, Sout, Dout, Lout>(
        &self,
        rhs0: &Tensor<T, Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1>,
        out: &mut Tensor<T, Sout, Dout, Contiguous, Lout, P>,
    ) where
        L: for<'a> Layout<'a, T>,
        Lrhs0: for<'a> Layout<'a, T>,
        Lrhs1: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self
            .opt_chunk_size()
            .min(rhs0.opt_chunk_size())
            .min(rhs1.opt_chunk_size());

        for (((chunk_self, chunck_rhs1), chunck_rhs2), chunk_out) in self
            .chunks(chunk_size)
            .zip(rhs0.chunks(chunk_size))
            .zip(rhs1.chunks(chunk_size))
            .zip(out.chunks_mut(chunk_size))
        {
            chunk_self
                .par_iter()
                .zip(chunck_rhs1.par_iter())
                .zip(chunck_rhs2.par_iter())
                .zip(chunk_out.par_iter_mut())
                .for_each(|(((x, y1), y2), z)| *z = x.placeholder(*y1, *y2));
        }
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd,
    mul_add<T=f32> in MulAdd,
)]
impl<T, S, C, L, P, Crhs0, Lrhs0, Prhs0, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, S, Static, Crhs0, Lrhs0, Prhs0>, &Tensor<T, S, Static, Crhs1, Lrhs1, Prhs1>> for &Tensor<T, S, Static, C, L, P>
where
    P: StaticAllocationPolicy<T, S>,
    L: for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, S, Static, Contiguous, P::Layout, P>;
    
    fn operation(
        self,
        rhs0: &Tensor<T, S, Static, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, S, Static, Crhs1, Lrhs1, Prhs1>,
    ) -> Self::Output {
        let mut out = Tensor::default();
        self.unchecked(rhs0, rhs1, &mut out);

        out
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd,
    mul_add<T=f32> in MulAdd,
)]
impl<T, S, C, L, P, Srhs0, Crhs0, Lrhs0, Prhs0, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, Srhs0, Dynamic, Crhs0, Lrhs0, Prhs0>, &Tensor<T, S, Static, Crhs1, Lrhs1, Prhs1>> for &Tensor<T, S, Static, C, L, P>
where
    S: Same<Srhs0>,
    <S as Same<Srhs0>>::Output: TRUE,
    P: StaticAllocationPolicy<T, S>,
    L: for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, S, Static, Contiguous, P::Layout, P>;
    
    fn operation(
        self,
        rhs0: &Tensor<T, Srhs0, Dynamic, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, S, Static, Crhs1, Lrhs1, Prhs1>,
    ) -> Self::Output {
        assert_shape_eq(self.shape(), rhs0.shape());

        let mut out = Tensor::default();
        self.unchecked(rhs0, rhs1, &mut out);

        out
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd,
    mul_add<T=f32> in MulAdd,
)]
impl<T, S, C, L, P, Crhs0, Lrhs0, Prhs0, Srhs1, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, S, Static, Crhs0, Lrhs0, Prhs0>, &Tensor<T, Srhs1, Dynamic, Crhs1, Lrhs1, Prhs1>> for &Tensor<T, S, Static, C, L, P>
where
    S: Same<Srhs1>,
    <S as Same<Srhs1>>::Output: TRUE,
    P: StaticAllocationPolicy<T, S>,
    L: for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, S, Static, Contiguous, P::Layout, P>;
    
    fn operation(
        self,
        rhs0: &Tensor<T, S, Static, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs1, Dynamic, Crhs1, Lrhs1, Prhs1>,
    ) -> Self::Output {
        assert_shape_eq(self.shape(), rhs1.shape());

        let mut out = Tensor::default();
        self.unchecked(rhs0, rhs1, &mut out);

        out
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd,
    mul_add<T=f32> in MulAdd,
)]
impl<T, S, C, L, P, Srhs, Crhs0, Lrhs0, Prhs0, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, Srhs, Static, Crhs0, Lrhs0, Prhs0>, &Tensor<T, Srhs, Static, Crhs1, Lrhs1, Prhs1>> for &Tensor<T, S, Dynamic, C, L, P>
where
    Srhs: Same<S>,
    <Srhs as Same<S>>::Output: TRUE,
    P: StaticAllocationPolicy<T, S>,
    L: for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, Srhs, Static, Contiguous, P::Layout, P>;
    
    fn operation(
        self,
        rhs0: &Tensor<T, Srhs, Static, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs, Static, Crhs1, Lrhs1, Prhs1>,
    ) -> Self::Output {
        assert_shape_eq(self.shape(), rhs0.shape());
        
        let mut out = Tensor::default();
        self.unchecked(rhs0, rhs1, &mut out);

        out
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd,
    mul_add<T=f32> in MulAdd,
)]
#[expand_impl(
    _a<D=Static, Drhs0=Dynamic, Drhs1=Dynamic, Sout=S>,
    _b<D=Dynamic, Drhs0=Static, Drhs1=Dynamic, Sout=Srhs0>,
    _c<D=Dynamic, Drhs0=Dynamic, Drhs1=Static, Sout=Srhs1>,
    _c<D=Dynamic, Drhs0=Dynamic, Drhs1=Dynamic, S: ReprShapeDyn<T, Srhs0>, <S as ReprShapeDyn<T, Srhs0>>::Output: ReprShapeDyn<T, Srhs1>, Sout=<<S as ReprShapeDyn<T, Srhs0>>::Output as ReprShapeDyn<T, Srhs1>>::Output>
)]
impl<T, S, C, L, P, Srhs0, Crhs0, Lrhs0, Prhs0, Srhs1, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0>, &Tensor<T, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1>> for &Tensor<T, S, D, C, L, P>
where
    S: Same<Srhs0> + Same<Srhs1>,
    <S as Same<Srhs0>>::Output: TRUE,
    <S as Same<Srhs1>>::Output: TRUE,
    P: StaticAllocationPolicy<T, Sout>,
    L: for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, Sout, Static, Contiguous, P::Layout, P>;
    
    fn operation(
        self,
        rhs0: &Tensor<T, Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1>,
    ) -> Self::Output {
        assert_shape_eq(self.shape(), rhs0.shape());
        assert_shape_eq(self.shape(), rhs1.shape());

        let mut out = Tensor::default();
        self.unchecked(rhs0, rhs1, &mut out);

        out
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAddDynamic,
    mul_add<T=f32> in MulAddDynamic,
)]
impl<T, S, C, L, P, Srhs0, Crhs0, Lrhs0, Prhs0, Srhs1, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, Srhs0, Dynamic, Crhs0, Lrhs0, Prhs0>, &Tensor<T, Srhs1, Dynamic, Crhs1, Lrhs1, Prhs1>> for &Tensor<T, S, Dynamic, C, L, P>
where
    S: Same<Srhs0> + Same<Srhs1> + ReprShapeDyn<T, Srhs0>,
    <S as ReprShapeDyn<T, Srhs0>>::Output: ReprShapeDyn<T, Srhs1>,
    <S as Same<Srhs0>>::Output: TRUE,
    <S as Same<Srhs1>>::Output: TRUE,
    P: DynamicAllocationPolicy<T>,
    L: for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{
    type Output = Tensor<T, <<S as ReprShapeDyn<T, Srhs0>>::Output as ReprShapeDyn<T, Srhs1>>::Output, Dynamic, Contiguous, P::Layout, P>;
    
    fn operation_dynamic(
        self,
        rhs0: &Tensor<T, Srhs0, Dynamic, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs1, Dynamic, Crhs1, Lrhs1, Prhs1>,
    ) -> Self::Output {
        assert_shape_eq(self.shape(), rhs0.shape());
        assert_shape_eq(self.shape(), rhs1.shape());

        let mut out = Tensor::alloc(self.shape());
        self.unchecked(rhs0, rhs1, &mut out);

        out
    }
}

#[expand_impl(
    add_assign<T: Send + Sync + Copy + AddAssign> as add,
    sub_assign<T: Send + Sync + Copy + SubAssign> as sub,
    mul_assign<T: Send + Sync + Copy + MulAssign> as mul,
    div_assign<T: Send + Sync + Copy + DivAssign> as div,
    rem_assign<T: Send + Sync + Copy + RemAssign> as rem,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked_<Srhs, Drhs, Crhs, Lrhs, Prhs>(&mut self, rhs: &Tensor<T, Srhs, Drhs, Crhs, Lrhs, Prhs>)
    where
        L: for<'a> LayoutMut<'a, T>,
        Lrhs: for<'a> Layout<'a, T>,
    {
        let chunk_size = rhs.opt_chunk_size();

        for (chunk_self, chunck_rhs) in self.chunks_mut(chunk_size).zip(rhs.chunks(chunk_size)) {
            chunk_self
                .par_iter_mut()
                .zip(chunck_rhs.par_iter())
                .for_each(|(x, y)| x.placeholder(*y));
        }
    }
}

#[expand_impl(
    atan2<T=f64>,
    copysign<T=f64>,
    div_euclid<T=f64>,
    max<T=f64>,
    min<T=f64>,
    rem_euclid<T=f64>,
    atan2<T=f32>,
    copysign<T=f32>,
    div_euclid<T=f32>,
    max<T=f32>,
    min<T=f32>,
    rem_euclid<T=f32>,
    div_euclid<T=u128>,
    rem_euclid<T=u128>,
    div_euclid<T=u64>,
    rem_euclid<T=u64>,
    div_euclid<T=u32>,
    rem_euclid<T=u32>,
    div_euclid<T=u16>,
    rem_euclid<T=u16>,
    div_euclid<T=u8>,
    rem_euclid<T=u8>,
    div_euclid<T=i128>,
    rem_euclid<T=i128>,
    div_euclid<T=i64>,
    rem_euclid<T=i64>,
    div_euclid<T=i32>,
    rem_euclid<T=i32>,
    div_euclid<T=i16>,
    rem_euclid<T=i16>,
    div_euclid<T=i8>,
    rem_euclid<T=i8>,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked_<Srhs, Drhs, Crhs, Lrhs, Prhs>(&mut self, rhs: &Tensor<T, Srhs, Drhs, Crhs, Lrhs, Prhs>)
    where
        L: for<'a> LayoutMut<'a, T>,
        Lrhs: for<'a> Layout<'a, T>,
    {
        let chunk_size = rhs.opt_chunk_size();

        for (chunk_self, chunck_rhs) in self.chunks_mut(chunk_size).zip(rhs.chunks(chunk_size)) {
            chunk_self
                .par_iter_mut()
                .zip(chunck_rhs.par_iter())
                .for_each(|(x, y)| *x = x.placeholder(*y));
        }
    }
}

#[expand_impl(
    add_assign<T: Send + Sync + Copy + AddAssign> as add in Add_,
    sub_assign<T: Send + Sync + Copy + SubAssign> as sub in Sub_,
    mul_assign<T: Send + Sync + Copy + MulAssign> as mul in Mul_,
    div_assign<T: Send + Sync + Copy + DivAssign> as div in Div_,
    rem_assign<T: Send + Sync + Copy + RemAssign> as rem in Rem_,
    atan2<T=f64> in Atan2_,
    copysign<T=f64> in Copysign_,
    div_euclid<T=f64> in DivEuclid_,
    max<T=f64> in Max_,
    min<T=f64> in Min_,
    rem_euclid<T=f64> in RemEuclid_,
    atan2<T=f32> in Atan2_,
    copysign<T=f32> in Copysign_,
    div_euclid<T=f32> in DivEuclid_,
    max<T=f32> in Max_,
    min<T=f32> in Min_,
    rem_euclid<T=f32> in RemEuclid_,
    div_euclid<T=u128> in DivEuclid_,
    rem_euclid<T=u128> in RemEuclid_,
    div_euclid<T=u64> in DivEuclid_,
    rem_euclid<T=u64> in RemEuclid_,
    div_euclid<T=u32> in DivEuclid_,
    rem_euclid<T=u32> in RemEuclid_,
    div_euclid<T=u16> in DivEuclid_,
    rem_euclid<T=u16> in RemEuclid_,
    div_euclid<T=u8> in DivEuclid_,
    rem_euclid<T=u8> in RemEuclid_,
    div_euclid<T=i128> in DivEuclid_,
    rem_euclid<T=i128> in RemEuclid_,
    div_euclid<T=i64> in DivEuclid_,
    rem_euclid<T=i64> in RemEuclid_,
    div_euclid<T=i32> in DivEuclid_,
    rem_euclid<T=i32> in RemEuclid_,
    div_euclid<T=i16> in DivEuclid_,
    rem_euclid<T=i16> in RemEuclid_,
    div_euclid<T=i8> in DivEuclid_,
    rem_euclid<T=i8> in RemEuclid_,
)]
impl<T, S, C, L, P, Crhs, Lrhs, Prhs> Operation<&Tensor<T, S, Static, Crhs, Lrhs, Prhs>> for Tensor<T, S, Static, C, L, P>
where
    L: for<'a> LayoutMut<'a, T>,
    Lrhs: for<'a> Layout<'a, T>,
{
    fn operation_(&mut self, rhs: &Tensor<T, S, Static, Crhs, Lrhs, Prhs>) {
        self.unchecked_(rhs);
    }
}

#[expand_impl(
    add_assign<T: Send + Sync + Copy + AddAssign> as add in Add_,
    sub_assign<T: Send + Sync + Copy + SubAssign> as sub in Sub_,
    mul_assign<T: Send + Sync + Copy + MulAssign> as mul in Mul_,
    div_assign<T: Send + Sync + Copy + DivAssign> as div in Div_,
    rem_assign<T: Send + Sync + Copy + RemAssign> as rem in Rem_,
    atan2<T=f64> in Atan2_,
    copysign<T=f64> in Copysign_,
    div_euclid<T=f64> in DivEuclid_,
    max<T=f64> in Max_,
    min<T=f64> in Min_,
    rem_euclid<T=f64> in RemEuclid_,
    atan2<T=f32> in Atan2_,
    copysign<T=f32> in Copysign_,
    div_euclid<T=f32> in DivEuclid_,
    max<T=f32> in Max_,
    min<T=f32> in Min_,
    rem_euclid<T=f32> in RemEuclid_,
    div_euclid<T=u128> in DivEuclid_,
    rem_euclid<T=u128> in RemEuclid_,
    div_euclid<T=u64> in DivEuclid_,
    rem_euclid<T=u64> in RemEuclid_,
    div_euclid<T=u32> in DivEuclid_,
    rem_euclid<T=u32> in RemEuclid_,
    div_euclid<T=u16> in DivEuclid_,
    rem_euclid<T=u16> in RemEuclid_,
    div_euclid<T=u8> in DivEuclid_,
    rem_euclid<T=u8> in RemEuclid_,
    div_euclid<T=i128> in DivEuclid_,
    rem_euclid<T=i128> in RemEuclid_,
    div_euclid<T=i64> in DivEuclid_,
    rem_euclid<T=i64> in RemEuclid_,
    div_euclid<T=i32> in DivEuclid_,
    rem_euclid<T=i32> in RemEuclid_,
    div_euclid<T=i16> in DivEuclid_,
    rem_euclid<T=i16> in RemEuclid_,
    div_euclid<T=i8> in DivEuclid_,
    rem_euclid<T=i8> in RemEuclid_,
)]
#[expand_impl(
    _a<D=Static, Drhs=Dynamic>,
    _b<D=Dynamic, Drhs=Static>,
    _c<D=Dynamic, Drhs=Dynamic>
)]
impl<T, S, C, L, P, Srhs, Crhs, Lrhs, Prhs> Operation<&Tensor<T, Srhs, Drhs, Crhs, Lrhs, Prhs>> for Tensor<T, S, D, C, L, P>
where
    S: Same<Srhs>,
    <S as Same<Srhs>>::Output: TRUE,
    L: for<'a> LayoutMut<'a, T> + for<'a> Layout<'a, T>,
    Lrhs: for<'a> Layout<'a, T>,
{
    fn operation_(&mut self, rhs: &Tensor<T, Srhs, Drhs, Crhs, Lrhs, Prhs>) {
        assert_shape_eq(self.shape(), rhs.shape());
        self.unchecked_(rhs);
    }
}

#[expand_impl(
    add<T: Send + Sync + Copy + Add<Output=T>>(T) as scal_add,
    sub<T: Send + Sync + Copy + Sub<Output=T>>(T) as scal_sub,
    mul<T: Send + Sync + Copy + Mul<Output=T>>(T) as scal_mul,
    div<T: Send + Sync + Copy + Div<Output=T>>(T) as scal_div,
    rem<T: Send + Sync + Copy + Rem<Output=T>>(T) as scal_rem,
    div_euclid<T=f64>(f64) as scal_div_euclid,
    max<T=f64>(f64) as scal_max,
    min<T=f64>(f64) as scal_min,
    powf<T=f64>(f64),
    rem_euclid<T=f64>(f64) as scal_rem_euclid,
    powi<T=f64>(i32),
    max<T=f32>(f32) as scal_max,
    min<T=f32>(f32) as scal_min,
    powf<T=f32>(f32),
    rem_euclid<T=f32>(f32) as scal_rem_euclid,
    powi<T=f32>(i32),
    div_euclid<T=u128>(u128) as scal_div_euclid,
    rem_euclid<T=u128>(u128) as scal_rem_euclid,
    div_euclid<T=u64>(u64) as scal_div_euclid,
    rem_euclid<T=u64>(u64) as scal_rem_euclid,
    div_euclid<T=u32>(u32) as scal_div_euclid,
    rem_euclid<T=u32>(u32) as scal_rem_euclid,
    div_euclid<T=u16>(u16) as scal_div_euclid,
    rem_euclid<T=u16>(u16) as scal_rem_euclid,
    div_euclid<T=u8>(u8) as scal_div_euclid,
    rem_euclid<T=u8>(u8) as scal_rem_euclid,
    div_euclid<T=i128>(i128) as scal_div_euclid,
    rem_euclid<T=i128>(i128) as scal_rem_euclid,
    div_euclid<T=i64>(i64) as scal_div_euclid,
    rem_euclid<T=i64>(i64) as scal_rem_euclid,
    div_euclid<T=i32>(i32) as scal_div_euclid,
    rem_euclid<T=i32>(i32) as scal_rem_euclid,
    div_euclid<T=i16>(i16) as scal_div_euclid,
    rem_euclid<T=i16>(i16) as scal_rem_euclid,
    div_euclid<T=i8>(i8) as scal_div_euclid,
    rem_euclid<T=i8>(i8) as scal_rem_euclid,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    pub fn operation_(&mut self, param: type0)
    where
        L: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();

        for chunk_self in self.chunks_mut(chunk_size) {
            chunk_self
                .par_iter_mut()
                .for_each(|x| *x = x.placeholder(param));
        }
    }
}

#[expand_impl(
    exp<T=f64>,
    exp2<T=f64>,
    exp_m1<T=f64>,
    ln<T=f64>,
    ln_1p<T=f64>,
    log2<T=f64>,
    log10<T=f64>,
    sin<T=f64>,
    cos<T=f64>,
    tan<T=f64>,
    sinh<T=f64>,
    cosh<T=f64>,
    tanh<T=f64>,
    asin<T=f64>,
    acos<T=f64>,
    atan<T=f64>,
    asinh<T=f64>,
    acosh<T=f64>,
    atanh<T=f64>,
    sqrt<T=f64>,
    cbrt<T=f64>,
    abs<T=f64>,
    signum<T=f64>,
    ceil<T=f64>,
    floor<T=f64>,
    round<T=f64>,
    recip<T=f64>,
    to_degrees<T=f64>,
    to_radians<T=f64>,
    exp<T=f32>,
    exp2<T=f32>,
    exp_m1<T=f32>,
    ln<T=f32>,
    ln_1p<T=f32>,
    log2<T=f32>,
    log10<T=f32>,
    sin<T=f32>,
    cos<T=f32>,
    tan<T=f32>,
    sinh<T=f32>,
    cosh<T=f32>,
    tanh<T=f32>,
    asin<T=f32>,
    acos<T=f32>,
    atan<T=f32>,
    asinh<T=f32>,
    acosh<T=f32>,
    atanh<T=f32>,
    sqrt<T=f32>,
    cbrt<T=f32>,
    abs<T=f32>,
    signum<T=f32>,
    ceil<T=f32>,
    floor<T=f32>,
    round<T=f32>,
    recip<T=f32>,
    to_degrees<T=f32>,
    to_radians<T=f32>,
    abs<T=i128>,
    signum<T=i128>,
    abs<T=i64>,
    signum<T=i64>,
    abs<T=i32>,
    signum<T=i32>,
    abs<T=i16>,
    signum<T=i16>,
    abs<T=i8>,
    signum<T=i8>,
    inv<T: Send + Sync + Copy + Ring + Div<Output = T>>,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    pub fn operation_(&mut self)
    where
        L: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();

        for chunk_self in self.chunks_mut(chunk_size) {
            chunk_self.par_iter_mut().for_each(|x| *x = x.placeholder());
        }
    }
}

#[expand_impl(
    mul_add<T=f64>(f64, f64) as scal_mul_add,
    mul_add<T=f32>(f32, f32) as scal_mul_add,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    pub fn operation_(&mut self, param0: type0, param1: type1)
    where
        L: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();

        for chunk_self in self.chunks_mut(chunk_size) {
            chunk_self
                .par_iter_mut()
                .for_each(|x| *x = x.placeholder(param0, param1));
        }
    }
}

#[expand_impl(
    mul_add<T=f64>,
    mul_add<T=f32>,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked_<Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1>(
        &mut self,
        rhs0: &Tensor<T, Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1>,
    ) where
        L: for<'a> LayoutMut<'a, T>,
        Lrhs0: for<'a> Layout<'a, T>,
        Lrhs1: for<'a> Layout<'a, T>,
    {
        let chunk_size = rhs0.opt_chunk_size().min(rhs1.opt_chunk_size());

        for ((chunk_self, chunck_rhs0), chunck_rhs1) in self
            .chunks_mut(chunk_size)
            .zip(rhs0.chunks(chunk_size))
            .zip(rhs1.chunks(chunk_size))
        {
            chunk_self
                .par_iter_mut()
                .zip(chunck_rhs0.par_iter())
                .zip(chunck_rhs1.par_iter())
                .for_each(|((x, y0), y1)| *x = x.placeholder(*y0, *y1));
        }
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd_,
    mul_add<T=f32> in MulAdd_,
)]
impl<T, S, C, L, P, Crhs0, Lrhs0, Prhs0, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, S, Static, Crhs0, Lrhs0, Prhs0>, &Tensor<T, S, Static, Crhs1, Lrhs1, Prhs1>> for Tensor<T, S, Static, C, L, P>
where
    L: for<'a> LayoutMut<'a, T> + for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{    
    fn operation_(
        &mut self,
        rhs0: &Tensor<T, S, Static, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, S, Static, Crhs1, Lrhs1, Prhs1>,
    )
    {
        self.unchecked_(rhs0, rhs1);
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd_,
    mul_add<T=f32> in MulAdd_,
)]
impl<T, S, C, L, P, Srhs0, Crhs0, Lrhs0, Prhs0, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, Srhs0, Dynamic, Crhs0, Lrhs0, Prhs0>, &Tensor<T, S, Static, Crhs1, Lrhs1, Prhs1>> for Tensor<T, S, Static, C, L, P>
where
    S: Same<Srhs0>,
    <S as Same<Srhs0>>::Output: TRUE,
    L: for<'a> LayoutMut<'a, T> + for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{    
    fn operation_(
        &mut self,
        rhs0: &Tensor<T, Srhs0, Dynamic, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, S, Static, Crhs1, Lrhs1, Prhs1>,
    )
    {
        assert_shape_eq(self.shape(), rhs0.shape());
        self.unchecked_(rhs0, rhs1);
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd_,
    mul_add<T=f32> in MulAdd_,
)]
impl<T, S, C, L, P, Crhs0, Lrhs0, Prhs0, Srhs1, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, S, Static, Crhs0, Lrhs0, Prhs0>, &Tensor<T, Srhs1, Dynamic, Crhs1, Lrhs1, Prhs1>> for Tensor<T, S, Static, C, L, P>
where
    S: Same<Srhs1>,
    <S as Same<Srhs1>>::Output: TRUE,
    L: for<'a> LayoutMut<'a, T> + for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{    
    fn operation_(
        &mut self,
        rhs0: &Tensor<T, S, Static, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs1, Dynamic, Crhs1, Lrhs1, Prhs1>,
    )
    {
        assert_shape_eq(self.shape(), rhs1.shape());
        self.unchecked_(rhs0, rhs1);
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd_,
    mul_add<T=f32> in MulAdd_,
)]
impl<T, S, C, L, P, Srhs, Crhs0, Lrhs0, Prhs0, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, Srhs, Static, Crhs0, Lrhs0, Prhs0>, &Tensor<T, Srhs, Static, Crhs1, Lrhs1, Prhs1>> for Tensor<T, S, Dynamic, C, L, P>
where
    Srhs: Same<S>,
    <Srhs as Same<S>>::Output: TRUE,
    L: for<'a> LayoutMut<'a, T> + for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{    
    fn operation_(
        &mut self,
        rhs0: &Tensor<T, Srhs, Static, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs, Static, Crhs1, Lrhs1, Prhs1>,
    )
    {
        assert_shape_eq(self.shape(), rhs0.shape());
        self.unchecked_(rhs0, rhs1);
    }
}

#[expand_impl(
    mul_add<T=f64> in MulAdd_,
    mul_add<T=f32> in MulAdd_,
)]
#[expand_impl(
    _a<D=Static, Drhs0=Dynamic, Drhs1=Dynamic>,
    _b<D=Dynamic, Drhs0=Static, Drhs1=Dynamic>,
    _c<D=Dynamic, Drhs0=Dynamic, Drhs1=Static>,
    _c<D=Dynamic, Drhs0=Dynamic, Drhs1=Dynamic>
)]
impl<T, S, D, C, L, P, Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1> Operation<&Tensor<T, Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0>, &Tensor<T, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1>> for Tensor<T, S, D, C, L, P>
where
    S: Same<Srhs0> + Same<Srhs1>,
    <S as Same<Srhs0>>::Output: TRUE,
    <S as Same<Srhs1>>::Output: TRUE,
    L: for<'a> LayoutMut<'a, T> + for<'a> Layout<'a, T>,
    Lrhs0: for<'a> Layout<'a, T>,
    Lrhs1: for<'a> Layout<'a, T>,
{    
    fn operation_(
        &mut self,
        rhs0: &Tensor<T, Srhs0, Drhs0, Crhs0, Lrhs0, Prhs0>,
        rhs1: &Tensor<T, Srhs1, Drhs1, Crhs1, Lrhs1, Prhs1>,
    )
    {
        assert_shape_eq(self.shape(), rhs0.shape());
        assert_shape_eq(self.shape(), rhs1.shape());
        self.unchecked_(rhs0, rhs1);
    }
}
