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

macro_rules! param_unary_math_op {
    ($f_unchecked:ident, $f:ident, $f_dyn:ident, $type:ty, $param_type:ty, $op:ident) => {
        fn $f_unchecked<Lout>(&self, param: $param_type, out: &mut Tensor<$type, S, Lout>)
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
                        .for_each(|(x, y)| *y = x.$op(param));
                }
        }

        pub fn $f(&self, param: $param_type) -> Tensor<$type, S, L::Default>
        where
            S: StaticShape,
            L: OpsDefaultOutput<$type, S>,
        {
            let mut out = Tensor::default();

            self.$f_unchecked(param, &mut out);

            out
        }

        pub fn $f_dyn(&self, param: $param_type) -> Tensor<$type, S, L::Alloc>
        where
            L: OpsAllocOutput<$type>,
        {
            let mut out = Tensor::alloc(self.shape());

            self.$f_unchecked(param, &mut out);

            out
        }
    };
}

macro_rules! binary_math_op {
    ($f_unchecked:ident, $f:ident, $f_static:ident, $f_dyn:ident, $type:ty, $op:ident) => {
        fn $f_unchecked<Srhs, Lrhs, Sout, Lout>(&self, other: &Tensor<$type, Srhs, Lrhs>, out: &mut Tensor<$type, Sout, Lout>)
        where
            Lrhs: for<'a> Layout<'a, $type>,
            Lout: for<'a> LayoutMut<'a, $type>,
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
                    .for_each(|((x, y), z)| *z = x.$op(*y));
            }
        }
        
        pub fn $f<Lrhs>(&self, other: &Tensor<$type, S, Lrhs>) -> Tensor<$type, S, L::Default>
        where
            S: StaticShape,
            L: OpsDefaultOutput<$type, S>,
            Lrhs: for<'a> Layout<'a, $type>,
        {
            let mut out = Tensor::default();
            
            self.$f_unchecked(other, &mut out);

            out
        }

        pub fn $f_static<Srhs, Lrhs>(&self, other: &Tensor<$type, Srhs, Lrhs>) -> Tensor<$type, <S as ReprShape<$type, Srhs>>::Output, L::Default>
        where
            S: Same<Srhs> + ReprShape<$type, Srhs>,
            <S as Same<Srhs>>::Output: TRUE,
            <S as ReprShape<$type, Srhs>>::Output: StaticShape,
            L: OpsDefaultOutput<$type, <S as ReprShape<$type, Srhs>>::Output>,
            Lrhs: for<'a> Layout<'a, $type>,
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

        pub fn $f_dyn<Srhs, Lrhs>(&self, other: &Tensor<$type, Srhs, Lrhs>) -> Tensor<$type, <S as ReprShapeDyn<$type, Srhs>>::Output, L::Alloc>
        where
            S: Same<Srhs> + ReprShapeDyn<$type, Srhs>,
            <S as Same<Srhs>>::Output: TRUE,
            L: OpsAllocOutput<$type>,
            Lrhs: for<'a> Layout<'a, $type>,
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

macro_rules! ternary_math_op {
    ($f_unchecked:ident, $f:ident, $f_static:ident, $f_dyn:ident, $type:ty, $op:ident) => {
        fn $f_unchecked<Srhs1, Lrhs1, Srhs2, Lrhs2, Sout, Lout>(&self, other1: &Tensor<$type, Srhs1, Lrhs1>, other2: &Tensor<$type, Srhs2, Lrhs2>, out: &mut Tensor<$type, Sout, Lout>)
        where
            Lrhs1: for<'a> Layout<'a, $type>,
            Lrhs2: for<'a> Layout<'a, $type>,
            Lout: for<'a> LayoutMut<'a, $type>,
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
                    .for_each(|(((x, y1), y2), z)| *z = x.$op(*y1, *y2));
            }
        }
        
        pub fn $f<Lrhs1, Lrhs2>(&self, other1: &Tensor<$type, S, Lrhs1>, other2: &Tensor<$type, S, Lrhs2>) -> Tensor<$type, S, L::Default>
        where
            S: StaticShape,
            L: OpsDefaultOutput<$type, S>,
            Lrhs1: for<'a> Layout<'a, $type>,
            Lrhs2: for<'a> Layout<'a, $type>,
        {
            let mut out = Tensor::default();
            
            self.$f_unchecked(other1, other2, &mut out);

            out
        }

        pub fn $f_static<Srhs1, Lrhs1, Srhs2, Lrhs2>(&self, other1: &Tensor<$type, Srhs1, Lrhs1>, other2: &Tensor<$type, Srhs2, Lrhs2>) -> Tensor<$type, <S as ReprShape<$type, <Srhs1 as ReprShape<$type, Srhs2>>::Output>>::Output, L::Default>
        where
            Srhs1: Same<Srhs2> + ReprShape<$type, Srhs2>,
            S: Same<Srhs1> + Same<Srhs2> + ReprShape<$type, <Srhs1 as ReprShape<$type, Srhs2>>::Output>,
            <S as Same<Srhs1>>::Output: TRUE,
            <S as Same<Srhs2>>::Output: TRUE,
            <Srhs1 as Same<Srhs2>>::Output: TRUE,
            <S as ReprShape<$type, <Srhs1 as ReprShape<$type, Srhs2>>::Output>>::Output: StaticShape,
            L: OpsDefaultOutput<$type, <S as ReprShape<$type, <Srhs1 as ReprShape<$type, Srhs2>>::Output>>::Output>,
            Lrhs1: for<'a> Layout<'a, $type>,
            Lrhs2: for<'a> Layout<'a, $type>,
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

            self.$f_unchecked(other1, other2, &mut out);

            out
        }

        pub fn $f_dyn<Srhs1, Lrhs1, Srhs2, Lrhs2>(&self, other1: &Tensor<$type, Srhs1, Lrhs1>, other2: &Tensor<$type, Srhs2, Lrhs2>) -> Tensor<$type, <S as ReprShapeDyn<$type, <Srhs1 as ReprShapeDyn<$type, Srhs2>>::Output>>::Output, L::Alloc>
        where
            Srhs1: Same<Srhs2> + ReprShapeDyn<$type, Srhs2>,
            S: Same<Srhs1> + Same<Srhs2> + ReprShapeDyn<$type, <Srhs1 as ReprShapeDyn<$type, Srhs2>>::Output>,
            <S as Same<Srhs1>>::Output: TRUE,
            <S as Same<Srhs2>>::Output: TRUE,
            <Srhs1 as Same<Srhs2>>::Output: TRUE,
            L: OpsAllocOutput<$type>,
            Lrhs1: for<'a> Layout<'a, $type>,
            Lrhs2: for<'a> Layout<'a, $type>,
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

            self.$f_unchecked(other1, other2, &mut out);

            out
        }
    };
}

macro_rules! two_param_unary_math_op {
    ($f_unchecked:ident, $f:ident, $f_dyn:ident, $type:ty, $param1_type:ty, $param2_type:ty, $op:ident) => {
        fn $f_unchecked<Lout>(&self, param1: $param1_type, param2: $param2_type, out: &mut Tensor<$type, S, Lout>)
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
                        .for_each(|(x, y)| *y = x.$op(param1, param2));
                }
        }

        pub fn $f(&self, param1: $param1_type, param2: $param2_type) -> Tensor<$type, S, L::Default>
        where
            S: StaticShape,
            L: OpsDefaultOutput<$type, S>,
        {
            let mut out = Tensor::default();

            self.$f_unchecked(param1, param2, &mut out);

            out
        }

        pub fn $f_dyn(&self, param1: $param1_type, param2: $param2_type) -> Tensor<$type, S, L::Alloc>
        where
            L: OpsAllocOutput<$type>,
        {
            let mut out = Tensor::alloc(self.shape());

            self.$f_unchecked(param1, param2, &mut out);

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

    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, f64, f64, div_euclid);
    param_unary_math_op!(scal_max_unchecked, scal_max, scla_max_dyn, f64, f64, max);
    param_unary_math_op!(scal_min_unchecked, scal_min, scal_min_dyn, f64, f64, min);
    param_unary_math_op!(powf_unchecked, powf, powf_dyn, f64, f64, powf);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, f64, f64, rem_euclid);
    param_unary_math_op!(powi_unchecked, powi, powi_dyn, f64, i32, powi);

    binary_math_op!(atan2_unchecked, atan2, atan2_static, atan2_dyn, f64, atan2);
    binary_math_op!(copysign_unchecked, copysign, copysign_static, copysign_dyn, f64, copysign);
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, f64, div_euclid);
    binary_math_op!(max_unchecked, max, max_static, max_dyn, f64, max);
    binary_math_op!(min_unchecked, min, min_static, min_dyn, f64, min);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, f64, rem_euclid);

    ternary_math_op!(mul_add_unchecked, mul_add, mul_add_static, mul_add_dyn, f64, mul_add);

    two_param_unary_math_op!(scal_mul_add_unchecked, scal_mul_add, scal_mul_add_dyn, f64, f64, f64, mul_add);
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

    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, f32, f32, div_euclid);
    param_unary_math_op!(scal_max_unchecked, scal_max, scla_max_dyn, f32, f32, max);
    param_unary_math_op!(scal_min_unchecked, scal_min, scal_min_dyn, f32, f32, min);
    param_unary_math_op!(powf_unchecked, powf, powf_dyn, f32, f32, powf);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, f32, f32, rem_euclid);
    param_unary_math_op!(powi_unchecked, powi, powi_dyn, f32, i32, powi);

    binary_math_op!(atan2_unchecked, atan2, atan2_static, atan2_dyn, f32, atan2);
    binary_math_op!(copysign_unchecked, copysign, copysign_static, copysign_dyn, f32, copysign);
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, f32, div_euclid);
    binary_math_op!(max_unchecked, max, max_static, max_dyn, f32, max);
    binary_math_op!(min_unchecked, min, min_static, min_dyn, f32, min);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, f32, rem_euclid);

    ternary_math_op!(mul_add_unchecked, mul_add, mul_add_static, mul_add_dyn, f32, mul_add);

    two_param_unary_math_op!(scal_mul_add_unchecked, scal_mul_add, scal_mul_add_dyn, f32, f32, f32, mul_add);
}

impl<S, L> Tensor<u128, S, L>
where
    L: for<'a> Layout<'a, u128>,
{
    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, u128, u128, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, u128, u128, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, u128, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, u128, rem_euclid);
}

impl<S, L> Tensor<u64, S, L>
where
    L: for<'a> Layout<'a, u64>,
{

    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, u64, u64, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, u64, u64, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, u64, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, u64, rem_euclid);
}

impl<S, L> Tensor<u32, S, L>
where
    L: for<'a> Layout<'a, u32>,
{
    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, u32, u32, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, u32, u32, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, u32, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, u32, rem_euclid);
}

impl<S, L> Tensor<u16, S, L>
where
    L: for<'a> Layout<'a, u16>,
{
    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, u16, u16, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, u16, u16, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, u16, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, u16, rem_euclid);
}

impl<S, L> Tensor<u8, S, L>
where
    L: for<'a> Layout<'a, u8>,
{
    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, u8, u8, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, u8, u8, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, u8, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, u8, rem_euclid);
}

impl<S, L> Tensor<i128, S, L>
where
    L: for<'a> Layout<'a, i128>,
{
    unary_math_op!(abs_unchecked, abs, abs_dyn, i128, abs);
    unary_math_op!(signum_unchecked, signum, signum_dyn, i128, signum);

    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, i128, i128, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, i128, i128, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, i128, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, i128, rem_euclid);
}

impl<S, L> Tensor<i64, S, L>
where
    L: for<'a> Layout<'a, i64>,
{
    unary_math_op!(abs_unchecked, abs, abs_dyn, i64, abs);
    unary_math_op!(signum_unchecked, signum, signum_dyn, i64, signum);

    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, i64, i64, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, i64, i64, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, i64, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, i64, rem_euclid);
}

impl<S, L> Tensor<i32, S, L>
where
    L: for<'a> Layout<'a, i32>,
{
    unary_math_op!(abs_unchecked, abs, abs_dyn, i32, abs);
    unary_math_op!(signum_unchecked, signum, signum_dyn, i32, signum);

    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, i32, i32, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, i32, i32, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, i32, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, i32, rem_euclid);
}

impl<S, L> Tensor<i16, S, L>
where
    L: for<'a> Layout<'a, i16>,
{
    unary_math_op!(abs_unchecked, abs, abs_dyn, i16, abs);
    unary_math_op!(signum_unchecked, signum, signum_dyn, i16, signum);

    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, i16, i16, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, i16, i16, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, i16, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, i16, rem_euclid);
}

impl<S, L> Tensor<i8, S, L>
where
    L: for<'a> Layout<'a, i8>,
{
    unary_math_op!(abs_unchecked, abs, abs_dyn, i8, abs);
    unary_math_op!(signum_unchecked, signum, signum_dyn, i8, signum);

    param_unary_math_op!(scal_div_euclid_unchecked, scal_div_euclid, scal_div_euclid_dyn, i8, i8, div_euclid);
    param_unary_math_op!(scal_rem_euclid_unchecked, scal_rem_euclid, scal_rem_euclid_dyn, i8, i8, rem_euclid);
    
    binary_math_op!(div_euclid_unchecked, div_euclid, div_euclid_static, div_euclid_dyn, i8, div_euclid);
    binary_math_op!(rem_euclid_unchecked, rem_euclid, rem_euclid_static, rem_euclid_dyn, i8, rem_euclid);
}
