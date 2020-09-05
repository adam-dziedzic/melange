use std::ops::*;
use rayon::prelude::*;
use super::tensor::Tensor;
use super::shape::{ReprShape, ReprShapeDyn, Same, StaticShape, TRUE};
use super::layout::{Layout, LayoutMut, OpsDefaultOutput, OpsAllocOutput};

macro_rules! binary_core_op {
    ($f_unchecked:ident, $f:ident, $f_static:ident, $f_dyn:ident, $bound:ident, $op:tt) => {
        fn $f_unchecked<Srhs, Lrhs, Sout, Lout>(&self, other: &Tensor<T, Srhs, Lrhs>, out: &mut Tensor<T, Sout, Lout>)
        where
            Lrhs: for<'a> Layout<'a, T>,
            Lout: for<'a> LayoutMut<'a, T>,
            T: $bound<Output=T>,
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
                    .for_each(|((x, y), z)| *z = *x $op *y);
            }
        }
        
        pub fn $f<Lrhs>(&self, other: &Tensor<T, S, Lrhs>) -> Tensor<T, S, L::Default>
        where
            S: StaticShape,
            L: OpsDefaultOutput<T, S>,
            Lrhs: for<'a> Layout<'a, T>,
            T: $bound<Output=T>,
        {
            let mut out = Tensor::default();
            
            self.$f_unchecked(other, &mut out);

            out
        }

        pub fn $f_static<Srhs, Lrhs>(&self, other: &Tensor<T, Srhs, Lrhs>) -> Tensor<T, <S as ReprShape<T, Srhs>>::Output, L::Default>
        where
            S: Same<Srhs> + ReprShape<T, Srhs>,
            <S as Same<Srhs>>::Output: TRUE,
            <S as ReprShape<T, Srhs>>::Output: StaticShape,
            L: OpsDefaultOutput<T, <S as ReprShape<T, Srhs>>::Output>,
            Lrhs: for<'a> Layout<'a, T>,
            T: $bound<Output=T>,
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

            self.$f_unchecked(other, &mut out);

            out
        }

        pub fn $f_dyn<Srhs, Lrhs>(&self, other: &Tensor<T, Srhs, Lrhs>) -> Tensor<T, <S as ReprShapeDyn<T, Srhs>>::Output, L::Alloc>
        where
            S: Same<Srhs> + ReprShapeDyn<T, Srhs>,
            <S as Same<Srhs>>::Output: TRUE,
            L: OpsAllocOutput<T>,
            Lrhs: for<'a> Layout<'a, T>,
            T: $bound<Output=T>,
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

            self.$f_unchecked(other, &mut out);

            out
        }
    };
}

macro_rules! scal_core_op {
    ($f_unchecked:ident, $f:ident, $f_dyn:ident, $bound:ident, $op:tt) => {
        fn $f_unchecked<Lout>(&self, scal: T, out: &mut Tensor<T, S, Lout>)
        where
            Lout: for<'a> LayoutMut<'a, T>,
            T: $bound<Output=T>,
        {
            let chunk_size = self.opt_chunk_size();
            
            for (chunk_self, chunk_out) in self
                .chunks(chunk_size)
                .zip(out.chunks_mut(chunk_size))
            {
                chunk_self
                    .par_iter()
                    .zip(chunk_out.par_iter_mut())
                    .for_each(|(x, y)| *y = *x $op scal);
            }
        }
        
        pub fn $f(&self, scal: T) -> Tensor<T, S, L::Default>
        where
            S: StaticShape,
            L: OpsDefaultOutput<T, S>,
            T: $bound<Output=T>,
        {
            let mut out = Tensor::default();

            self.$f_unchecked(scal, &mut out);

            out
        }

        pub fn $f_dyn(&self, scal: T) -> Tensor<T, S, L::Alloc>
        where
            L: OpsAllocOutput<T>,
            T: $bound<Output=T>,
        {
            let mut out = Tensor::alloc(self.shape());

            self.$f_unchecked(scal, &mut out);

            out
        }
    };
}

macro_rules! unary_math_op {
    ($f_unchecked:ident, $f:ident, $f_dyn:ident, $type:ty, $op:ident) => {
        fn $f_unchecked<Lout>(&self, out: &mut Tensor<$type, S, Lout>)
        where
            Lout: for<'a> LayoutMut<'a, $type>,
        {
            let chunk_size = self.opt_chunk_size();
                
                for (chunk_self, chunk_out) in self
                    .chunks(chunk_size)
                    .zip(out.chunks_mut(chunk_size))
                {
                    chunk_self
                        .par_iter()
                        .zip(chunk_out.par_iter_mut())
                        .for_each(|(x, y)| *y = x.$op());
                }
        }

        pub fn $f(&self) -> Tensor<$type, S, L::Default>
        where
            S: StaticShape,
            L: OpsDefaultOutput<$type, S>,
        {
            let mut out = Tensor::default();

            self.$f_unchecked(&mut out);

            out
        }

        pub fn $f_dyn(&self) -> Tensor<$type, S, L::Alloc>
        where
            L: OpsAllocOutput<$type>,
        {
            let mut out = Tensor::alloc(self.shape());

            self.$f_unchecked(&mut out);

            out
        }
    };
}

impl<T, S, L> Tensor<T, S, L>
where
    L: for<'a> Layout<'a, T>,
    T: Send + Sync + Copy,
{
    binary_core_op!(add_unchecked, add, add_static, add_dyn, Add, +);
    binary_core_op!(sub_unchecked, sub, sub_static, sub_dyn, Sub, -);
    binary_core_op!(mul_unchecked, mul, mul_static, mul_dyn, Mul, *);
    binary_core_op!(div_unchecked, div, div_static, div_dyn, Div, /);
    binary_core_op!(rem_unchecked, rem, rem_static, rem_dyn, Rem, %);

    scal_core_op!(scal_add_unchecked, scal_add, scal_add_dyn, Add, +);
    scal_core_op!(scal_sub_unchecked, scal_sub, scal_sub_dyn, Sub, -);
    scal_core_op!(scal_mul_unchecked, scal_mul, scal_mul_dyn, Mul, *);
    scal_core_op!(scal_div_unchecked, scal_div, scal_div_dyn, Div, /);
    scal_core_op!(scal_rem_unchecked, scal_rem, scal_rem_dyn, Rem, %);
}

impl<S, L> Tensor<f64, S, L>
where
    L: for<'a> Layout<'a, f64>,
{
    unary_math_op!(exp_unchecked, exp, exp_dyn, f64, exp);
    unary_math_op!(exp2_unchecked, exp2, exp2_dyn, f64, exp2);
    unary_math_op!(exp_m1_unchecked, exp_m1, exp_m1_dyn, f64, exp_m1);
    unary_math_op!(ln_unchecked, ln, ln_dyn, f64, ln);
    unary_math_op!(ln_1p_unchecked, ln_1p, ln_1p_dyn, f64, ln_1p);
    unary_math_op!(log2_unchecked, log2, log2_dyn, f64, log2);
    unary_math_op!(log10_unchecked, log10, log10_dyn, f64, log10);
    unary_math_op!(sin_unchecked, sin, sin_dyn, f64, sin);
    unary_math_op!(cos_unchecked, cos, cos_dyn, f64, cos);
    unary_math_op!(tan_unchecked, tan, tan_dyn, f64, tan);
    unary_math_op!(sinh_unchecked, sinh, sinh_dyn, f64, sinh);
    unary_math_op!(cosh_unchecked, cosh, cosh_dyn, f64, cosh);
    unary_math_op!(tanh_unchecked, tanh, tanh_dyn, f64, tanh);
    unary_math_op!(asin_unchecked, asin, asin_dyn, f64, asin);
    unary_math_op!(acos_unchecked, acos, acos_dyn, f64, acos);
    unary_math_op!(atan_unchecked, atan, atan_dyn, f64, atan);
    unary_math_op!(asinh_unchecked, asinh, asinh_dyn, f64, asinh);
    unary_math_op!(acosh_unchecked, acosh, acosh_dyn, f64, acosh);
    unary_math_op!(atanh_unchecked, atanh, atanh_dyn, f64, atanh);
    unary_math_op!(sqrt_unchecked, sqrt, sqrt_dyn, f64, sqrt);
    unary_math_op!(cbrt_unchecked, cbrt, cbrt_dyn, f64, cbrt);
    unary_math_op!(abs_unchecked, abs, abs_dyn, f64, abs);
    unary_math_op!(signum_unchecked, signum, signum_dyn, f64, signum);
    unary_math_op!(ceil_unchecked, ceil, ceil_dyn, f64, ceil);
    unary_math_op!(floor_unchecked, floor, floor_dyn, f64, floor);
    unary_math_op!(round_unchecked, round, round_dyn, f64, round);
    unary_math_op!(recip_unchecked, recip, recip_dyn, f64, recip);
    unary_math_op!(to_degrees_unchecked, to_degrees, to_degrees_dyn, f64, to_degrees);
    unary_math_op!(to_radians_unchecked, to_radians, to_radians_dyn, f64, to_radians);
}

impl<S, L> Tensor<f32, S, L>
where
    L: for<'a> Layout<'a, f32>,
{
    unary_math_op!(exp_unchecked, exp, exp_dyn, f32, exp);
    unary_math_op!(exp2_unchecked, exp2, exp2_dyn, f32, exp2);
    unary_math_op!(exp_m1_unchecked, exp_m1, exp_m1_dyn, f32, exp_m1);
    unary_math_op!(ln_unchecked, ln, ln_dyn, f32, ln);
    unary_math_op!(ln_1p_unchecked, ln_1p, ln_1p_dyn, f32, ln_1p);
    unary_math_op!(log2_unchecked, log2, log2_dyn, f32, log2);
    unary_math_op!(log10_unchecked, log10, log10_dyn, f32, log10);
    unary_math_op!(sin_unchecked, sin, sin_dyn, f32, sin);
    unary_math_op!(cos_unchecked, cos, cos_dyn, f32, cos);
    unary_math_op!(tan_unchecked, tan, tan_dyn, f32, tan);
    unary_math_op!(sinh_unchecked, sinh, sinh_dyn, f32, sinh);
    unary_math_op!(cosh_unchecked, cosh, cosh_dyn, f32, cosh);
    unary_math_op!(tanh_unchecked, tanh, tanh_dyn, f32, tanh);
    unary_math_op!(asin_unchecked, asin, asin_dyn, f32, asin);
    unary_math_op!(acos_unchecked, acos, acos_dyn, f32, acos);
    unary_math_op!(atan_unchecked, atan, atan_dyn, f32, atan);
    unary_math_op!(asinh_unchecked, asinh, asinh_dyn, f32, asinh);
    unary_math_op!(acosh_unchecked, acosh, acosh_dyn, f32, acosh);
    unary_math_op!(atanh_unchecked, atanh, atanh_dyn, f32, atanh);
    unary_math_op!(sqrt_unchecked, sqrt, sqrt_dyn, f32, sqrt);
    unary_math_op!(cbrt_unchecked, cbrt, cbrt_dyn, f32, cbrt);
    unary_math_op!(abs_unchecked, abs, abs_dyn, f32, abs);
    unary_math_op!(signum_unchecked, signum, signum_dyn, f32, signum);
    unary_math_op!(ceil_unchecked, ceil, ceil_dyn, f32, ceil);
    unary_math_op!(floor_unchecked, floor, floor_dyn, f32, floor);
    unary_math_op!(round_unchecked, round, round_dyn, f32, round);
    unary_math_op!(recip_unchecked, recip, recip_dyn, f32, recip);
    unary_math_op!(to_degrees_unchecked, to_degrees, to_degrees_dyn, f32, to_degrees);
    unary_math_op!(to_radians_unchecked, to_radians, to_radians_dyn, f32, to_radians);
}
