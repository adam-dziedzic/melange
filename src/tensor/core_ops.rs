use std::ops::*;
use rayon::prelude::*;
use super::tensor::Tensor;
use super::shape::{ReprShape, Same, StaticShape, TRUE};
use super::layout::{Layout, LayoutMut};

macro_rules! binary_core_op {
    ($f1:ident, $f2:ident, $f3:ident, $bound:ident, $op:tt) => {
        fn $f1<Srhs, Lrhs, Sout, Lout>(&self, other: &Tensor<T, Srhs, Lrhs>) -> Tensor<T, Sout, Lout>
        where
            Lrhs: for<'a> Layout<'a, T>,
            Lout: for<'a> LayoutMut<'a, T> + Default,
            T: $bound<Output=T>,
        {
            let chunk_size = self.opt_chunk_size().min(other.opt_chunk_size());

            let mut out: Tensor<T, Sout, Lout> = Tensor::default();

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

            out
        }
        
        pub fn $f2<Lrhs, Lout>(&self, other: &Tensor<T, S, Lrhs>) -> Tensor<T, S, Lout>
        where
            S: StaticShape,
            Lrhs: for<'a> Layout<'a, T>,
            Lout: for<'a> LayoutMut<'a, T> + Default,
            T: $bound<Output=T>,
        {
            self.$f1(other)
        }

        pub fn $f3<Srhs, Lrhs, Lout>(&self, other: &Tensor<T, Srhs, Lrhs>) -> Tensor<T, <S as ReprShape<T, Srhs>>::Output, Lout>
        where
            S: Same<Srhs> + ReprShape<T, Srhs>,
            <S as Same<Srhs>>::Output: TRUE,
            Lrhs: for<'a> Layout<'a, T>,
            Lout: for<'a> LayoutMut<'a, T> + Default,
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

            self.$f1(other)
        }
    };
}


impl<T, S, L> Tensor<T, S, L>
where
    L: for<'a> Layout<'a, T>,
    T: Send + Sync + Copy,
{
    binary_core_op!(add_unchecked, add, add_dyn, Add, +);
    binary_core_op!(sub_unchecked, sub, sub_dyn, Sub, -);
    binary_core_op!(mul_unchecked, mul, mul_dyn, Mul, *);
    binary_core_op!(div_unchecked, div, div_dyn, Div, /);
    binary_core_op!(rem_unchecked, rem, rem_dyn, Rem, %);


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