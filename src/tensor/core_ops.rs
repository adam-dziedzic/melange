use std::ops::*;
use rayon::prelude::*;
use road_ai_macros::expand_operations;
use super::tensor::Tensor;
use super::shape::{ReprShape, ReprShapeDyn, Same, StaticShape, TRUE};
use super::layout::{Layout, LayoutMut};
use super::allocation_policy::{StaticAllocationPolicy, DynamicAllocationPolicy};

#[expand_operations(
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
impl<T, S, L, P> Tensor<T, S, L, P>
where
    L: for <'a> Layout<'a, T>,
{
    #[inline]
    fn unchecked<Srhs, Lrhs, Prhs, Sout, Lout>(&self, other: &Tensor<T, Srhs, Lrhs, Prhs>, out: &mut Tensor<T, Sout, Lout, P>)
    where
        Lrhs: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size().min(other.opt_chunk_size());

        for ((chunk_self, chunk_other), chunk_out) in self
            .chunks(chunk_size)
            .zip(other.chunks(chunk_size))
            .zip(out.chunks_mut(chunk_size))
        {
            chunk_self.par_iter()
                .zip(chunk_other.par_iter())
                .zip(chunk_out.par_iter_mut())
                .for_each(|((x, y), z)| *z = x.placeholder(*y));
        }
    }
    
    pub fn operation<Lrhs, Prhs>(&self, other: &Tensor<T, S, Lrhs, Prhs>) -> Tensor<T, S, P::Layout, P>
    where
        S: StaticShape,
        P: StaticAllocationPolicy<T, S>,
        Lrhs: for<'a> Layout<'a, T>,
    {
        let mut out = Tensor::default();
        
        self.unchecked(other, &mut out);

        out
    }

    pub fn coerce<Srhs, Lrhs, Prhs>(&self, other: &Tensor<T, Srhs, Lrhs, Prhs>) -> Tensor<T, <S as ReprShape<T, Srhs>>::Output, P::Layout, P>
    where
        S: Same<Srhs> + ReprShape<T, Srhs>,
        <S as Same<Srhs>>::Output: TRUE,
        P: StaticAllocationPolicy<T, <S as ReprShape<T, Srhs>>::Output>,
        Lrhs: for<'a> Layout<'a, T>,
    {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape,
            other_shape,
            "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
            self_shape,
            other_shape
        );

        let mut out = Tensor::default();

        self.unchecked(other, &mut out);

        out
    }

    pub fn dynamic<Srhs, Lrhs, Prhs>(&self, other: &Tensor<T, Srhs, Lrhs, Prhs>) -> Tensor<T, <S as ReprShapeDyn<T, Srhs>>::Output, P::Layout, P>
    where
        S: Same<Srhs> + ReprShapeDyn<T, Srhs>,
        <S as Same<Srhs>>::Output: TRUE,
        P: DynamicAllocationPolicy<T>,
        Lrhs: for<'a> Layout<'a, T>,
    {
        let self_shape = self.shape();
        let other_shape = other.shape();
        assert_eq!(
            self_shape,
            other_shape,
            "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
            self_shape,
            other_shape
        );

        let mut out = Tensor::alloc(self_shape);

        self.unchecked(other, &mut out);

        out
    }
}

#[expand_operations(
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
impl<T, S, L, P> Tensor<T, S, L, P>
where
    L: for <'a> Layout<'a, T>,
{
    #[inline]
    fn unchecked<Lout>(&self, param: type0, out: &mut Tensor<T, S, Lout, P>)
    where
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();
            
            for (chunk_self, chunk_out) in self
                .chunks(chunk_size)
                .zip(out.chunks_mut(chunk_size))
            {
                chunk_self
                    .par_iter()
                    .zip(chunk_out.par_iter_mut())
                    .for_each(|(x, y)| *y = x.placeholder(param));
            }
    }

    pub fn operation(&self, param: type0) -> Tensor<T, S, P::Layout, P>
    where
        S: StaticShape,
        P: StaticAllocationPolicy<T, S>,
    {
        let mut out = Tensor::default();

        self.unchecked(param, &mut out);

        out
    }

    pub fn dynamic(&self, param: type0) -> Tensor<T, S, P::Layout, P>
    where
        P: DynamicAllocationPolicy<T>,
    {
        let mut out = Tensor::alloc(self.shape());

        self.unchecked(param, &mut out);

        out
    }
}

#[expand_operations(
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
)]
impl<T, S, L, P> Tensor<T, S, L, P>
where
    L: for <'a> Layout<'a, T>,
{
    #[inline]
    fn unchecked<Lout>(&self, out: &mut Tensor<T, S, Lout, P>)
    where
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();
            
            for (chunk_self, chunk_out) in self
                .chunks(chunk_size)
                .zip(out.chunks_mut(chunk_size))
            {
                chunk_self
                    .par_iter()
                    .zip(chunk_out.par_iter_mut())
                    .for_each(|(x, y)| *y = x.placeholder());
            }
    }

    pub fn operation(&self) -> Tensor<T, S, P::Layout, P>
    where
        S: StaticShape,
        P: StaticAllocationPolicy<T, S>,
    {
        let mut out = Tensor::default();

        self.unchecked(&mut out);

        out
    }

    pub fn dynamic(&self) -> Tensor<T, S, P::Layout, P>
    where
        P: DynamicAllocationPolicy<T>,
    {
        let mut out = Tensor::alloc(self.shape());

        self.unchecked(&mut out);

        out
    }
}

#[expand_operations(
    mul_add<T=f64>(f64, f64) as scal_mul_add,
    mul_add<T=f32>(f32, f32) as scal_mul_add,
)]
impl<T, S, L, P> Tensor<T, S, L, P>
where
    L: for <'a> Layout<'a, T>,
{
    #[inline]
    fn unchecked<Lout>(&self, param1: type0, param2: type1, out: &mut Tensor<T, S, Lout, P>)
    where
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self.opt_chunk_size();
            
            for (chunk_self, chunk_out) in self
                .chunks(chunk_size)
                .zip(out.chunks_mut(chunk_size))
            {
                chunk_self
                    .par_iter()
                    .zip(chunk_out.par_iter_mut())
                    .for_each(|(x, y)| *y = x.placeholder(param1, param2));
            }
    }

    pub fn operation(&self, param1: type0, param2: type1) -> Tensor<T, S, P::Layout, P>
    where
        S: StaticShape,
        P: StaticAllocationPolicy<T, S>,
    {
        let mut out = Tensor::default();

        self.unchecked(param1, param2, &mut out);

        out
    }

    pub fn dynamic(&self, param1: type0, param2: type1) -> Tensor<T, S, P::Layout, P>
    where
        P: DynamicAllocationPolicy<T>,
    {
        let mut out = Tensor::alloc(self.shape());

        self.unchecked(param1, param2, &mut out);

        out
    }
}

#[expand_operations(
    mul_add<T=f64>,
    mul_add<T=f32>,
)]
impl<T, S, L, P> Tensor<T, S, L, P>
where
    L: for <'a> Layout<'a, T>,
{
    #[inline]
    fn unchecked<Srhs1, Lrhs1, Prhs1, Srhs2, Lrhs2, Prhs2, Sout, Lout>(&self, other1: &Tensor<T, Srhs1, Lrhs1, Prhs1>, other2: &Tensor<T, Srhs2, Lrhs2, Prhs2>, out: &mut Tensor<T, Sout, Lout, P>)
    where
        Lrhs1: for<'a> Layout<'a, T>,
        Lrhs2: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let chunk_size = self
            .opt_chunk_size()
            .min(other1.opt_chunk_size())
            .min(other2.opt_chunk_size());

        for (((chunk_self, chunk_other1), chunk_other2), chunk_out) in self
            .chunks(chunk_size)
            .zip(other1.chunks(chunk_size))
            .zip(other2.chunks(chunk_size))
            .zip(out.chunks_mut(chunk_size))
        {
            chunk_self.par_iter()
                .zip(chunk_other1.par_iter())
                .zip(chunk_other2.par_iter())
                .zip(chunk_out.par_iter_mut())
                .for_each(|(((x, y1), y2), z)| *z = x.placeholder(*y1, *y2));
        }
    }
    
    pub fn operation<Lrhs1, Prhs1, Lrhs2, Prhs2>(&self, other1: &Tensor<T, S, Lrhs1, Prhs1>, other2: &Tensor<T, S, Lrhs2, Prhs2>) -> Tensor<T, S, P::Layout, P>
    where
        S: StaticShape,
        P: StaticAllocationPolicy<T, S>,
        Lrhs1: for<'a> Layout<'a, T>,
        Lrhs2: for<'a> Layout<'a, T>,
    {
        let mut out = Tensor::default();
        
        self.unchecked(other1, other2, &mut out);

        out
    }

    pub fn coerce<Srhs1, Lrhs1, Prhs1, Srhs2, Lrhs2, Prhs2>(&self, other1: &Tensor<T, Srhs1, Lrhs1, Prhs1>, other2: &Tensor<T, Srhs2, Lrhs2, Prhs2>) -> Tensor<T, <S as ReprShape<T, <Srhs1 as ReprShapeDyn<T, Srhs2>>::Output>>::Output, P::Layout, P>
    where
        S: Same<Srhs1> + Same<Srhs2> + ReprShape<T, <Srhs1 as ReprShapeDyn<T, Srhs2>>::Output>,
        <S as Same<Srhs1>>::Output: TRUE,
        <S as Same<Srhs2>>::Output: TRUE,
        P: StaticAllocationPolicy<T, <S as ReprShape<T, <Srhs1 as ReprShapeDyn<T, Srhs2>>::Output>>::Output>,
        Srhs1: Same<Srhs2> + ReprShapeDyn<T, Srhs2>,
        <Srhs1 as Same<Srhs2>>::Output: TRUE,
        Lrhs1: for<'a> Layout<'a, T>,
        Lrhs2: for<'a> Layout<'a, T>,
    {
        let self_shape = self.shape();
        let other1_shape = other1.shape();
        let other2_shape = other2.shape();
        assert_eq!(
            self_shape,
            other1_shape,
            "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
            self_shape,
            other1_shape
        );
        assert_eq!(
            self_shape,
            other2_shape,
            "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
            self_shape,
            other2_shape
        );

        let mut out = Tensor::default();

        self.unchecked(other1, other2, &mut out);

        out
    }

    pub fn dynamic<Srhs1, Lrhs1, Prhs1, Srhs2, Lrhs2, Prhs2>(&self, other1: &Tensor<T, Srhs1, Lrhs1, Prhs1>, other2: &Tensor<T, Srhs2, Lrhs2, Prhs2>) -> Tensor<T, <S as ReprShapeDyn<T, <Srhs1 as ReprShapeDyn<T, Srhs2>>::Output>>::Output, P::Layout, P>
    where
        S: Same<Srhs1> + Same<Srhs2> + ReprShapeDyn<T, <Srhs1 as ReprShapeDyn<T, Srhs2>>::Output>,
        <S as Same<Srhs1>>::Output: TRUE,
        <S as Same<Srhs2>>::Output: TRUE,
        P: DynamicAllocationPolicy<T>,
        Srhs1: Same<Srhs2> + ReprShapeDyn<T, Srhs2>,
        <Srhs1 as Same<Srhs2>>::Output: TRUE,
        Lrhs1: for<'a> Layout<'a, T>,
        Lrhs2: for<'a> Layout<'a, T>,
    {
        let self_shape = self.shape();
        let other1_shape = other1.shape();
        let other2_shape = other2.shape();
        assert_eq!(
            self_shape,
            other1_shape,
            "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
            self_shape,
            other1_shape
        );
        assert_eq!(
            self_shape,
            other2_shape,
            "Tensors must have same shape, got {:?} and {:?}. The use of static shapes (compile-time-known type-level shapes) is strongly recommended.",
            self_shape,
            other2_shape
        );

        let mut out = Tensor::alloc(self_shape);

        self.unchecked(other1, other2, &mut out);

        out
    }
}