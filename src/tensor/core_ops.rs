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


    // fn scal_add<Lout>(&self, scal: T) -> Tensor<T, S, Lout>
    // where
    //     S: StaticShape,
    //     Lout: for<'a> LayoutMut<'a, T> + Default,
    //     T: Add<Output=T>,
    // {
    //     let chunk_size = self.opt_chunk_size();

    //     let mut out: Tensor<T, S, Lout> = Tensor::default();

    //     for (chunk_self, chunk_out) in self
    //         .chunks(chunk_size)
    //         .zip(out.chunks_mut(chunk_size))
    //     {
    //         chunk_self
    //             .par_iter()
    //             .zip(chunk_out.par_iter_mut())
    //             .for_each(|(x, y)| *y = *x + scal);
    //     }

    //     out
    // }

    // fn scal_add_dyn<Sout, Lout>(&self, scal: T) -> Tensor<T, Sout, Lout>
    // where
    //     S: StaticShape,
    //     Lout: for<'a> LayoutMut<'a, T> + Default,
    //     T: Add<Output=T>,
    // {
    //     let chunk_size = self.opt_chunk_size();

    //     let mut out: Tensor<T, Sout, Lout> = Tensor::default();

    //     for (chunk_self, chunk_out) in self
    //         .chunks(chunk_size)
    //         .zip(out.chunks_mut(chunk_size))
    //     {
    //         chunk_self
    //             .par_iter()
    //             .zip(chunk_out.par_iter_mut())
    //             .for_each(|(x, y)| *y = *x + scal);
    //     }

    //     out
    // }

    //fn scal
}