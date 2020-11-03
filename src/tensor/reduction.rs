//! `reductions` contains mathematical operations that aply to one
//! tensor and output a tensor with the same shape except for one
//! dimension that is reduced to 1.
//! This covers sum, product, max and min over a certain axis.
//!
//! Like core ops, these methods heavily use the chunks feature
//! of the `Layout` trait to parallelize.
//!
//! To avoid code duplication, this module relies on the
//! `expand_operations` procedural macro from the `melange_macro` crate.

use super::allocation_policy::{DynamicAllocationPolicy, StaticAllocationPolicy};
use super::layout::{Layout, LayoutMut};
use super::shape::{At, Reduction, ReductionOptChunckSize, Static, Dynamic};
use super::tensor::Tensor;
use super::transpose_policy::Contiguous;
use rayon::prelude::*;
use melange_macros::expand_impl;
use typenum::Unsigned;
use std::ops::*;

#[expand_impl(
    add_assign<T: Send + Sync + Copy + AddAssign> as sum,
    mul_assign<T: Send + Sync + Copy + MulAssign> as prod,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked<Sout, Dout, Lout>(
        &self,
        chunk_size_in: usize,
        chunk_size_out: usize,
        mid_loop_num: usize,
        inner_loop_num: usize,
        out: &mut Tensor<T, Sout, Dout, Contiguous, Lout, P>,
    ) where
        L: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let mut in_iter = self.chunks(chunk_size_in);
        for chunk_o in out.chunks_mut(chunk_size_out) {
            for _ in 0..mid_loop_num {
                for j in 0..inner_loop_num {
                    let chunk_i = in_iter.next().unwrap();
                    chunk_o
                        .par_iter_mut()
                        .skip(j * chunk_size_in)
                        .zip(chunk_i.par_iter())
                        .for_each(|(o, i)| o.placeholder(*i));
                }
            }
        }
    }
}

#[expand_impl(
    max<T=f64> as reduce_max,
    min<T=f64> as reduce_min,
    max<T=f32> as reduce_max,
    min<T=f32> as reduce_min,
)]
impl<T, S, D, C, L, P> Tensor<T, S, D, C, L, P> {
    #[inline]
    fn operation_unchecked<Sout, Dout, Lout>(
        &self,
        chunk_size_in: usize,
        chunk_size_out: usize,
        mid_loop_num: usize,
        inner_loop_num: usize,
        out: &mut Tensor<T, Sout, Dout, Contiguous, Lout, P>,
    ) where
        L: for<'a> Layout<'a, T>,
        Lout: for<'a> LayoutMut<'a, T>,
    {
        let mut in_iter = self.chunks(chunk_size_in);
        for chunk_o in out.chunks_mut(chunk_size_out) {
            for _ in 0..mid_loop_num {
                for j in 0..inner_loop_num {
                    let chunk_i = in_iter.next().unwrap();
                    chunk_o
                        .par_iter_mut()
                        .skip(j * chunk_size_in)
                        .zip(chunk_i.par_iter())
                        .for_each(|(o, i)| *o = o.placeholder(*i));
                }
            }
        }
    }
}

#[expand_impl(
    add_assign<T: Send + Sync + Copy + AddAssign> as sum,
    mul_assign<T: Send + Sync + Copy + MulAssign> as prod,
    max<T=f64> as reduce_max,
    min<T=f64> as reduce_min,
    max<T=f32> as reduce_max,
    min<T=f32> as reduce_min,
)]
impl<T, S, C, L, P> Tensor<T, S, Static, C, L, P> {
    pub fn operation<Ax>(&self) -> Tensor<T, <S as Reduction<Ax>>::Output, Static, Contiguous, P::Layout, P>
    where
        S: Reduction<Ax> + ReductionOptChunckSize<T, Ax> + At<Ax>,
        L: for<'a> Layout<'a, T>,
        P: StaticAllocationPolicy<T, <S as Reduction<Ax>>::Output>,
    {
        let chunk_size_out = <<S as ReductionOptChunckSize<T, Ax>>::Output as Unsigned>::USIZE;
        let chunk_size_in = self.opt_chunk_size().min(chunk_size_out);

        let inner_loop_num = chunk_size_out / chunk_size_in;
        let mid_loop_num = <<S as At<Ax>>::Output as Unsigned>::USIZE;

        let mut out: Tensor<T, <S as Reduction<Ax>>::Output, Static, Contiguous, P::Layout, P> =
            Tensor::default();

        self.unchecked(
            chunk_size_in,
            chunk_size_out,
            mid_loop_num,
            inner_loop_num,
            &mut out,
        );
        out
    }
}

#[expand_impl(
    add_assign<T: Send + Sync + Copy + AddAssign> as sum,
    mul_assign<T: Send + Sync + Copy + MulAssign> as prod,
    max<T=f64> as reduce_max,
    min<T=f64> as reduce_min,
    max<T=f32> as reduce_max,
    min<T=f32> as reduce_min,
)]
impl<T, S, C, L, P> Tensor<T, S, Dynamic, C, L, P> {
    pub fn operation<Ax>(&self) -> Tensor<T, <S as Reduction<Ax>>::Output, Dynamic, Contiguous, P::Layout, P>
    where
        S: Reduction<Ax>,
        P: DynamicAllocationPolicy<T>,
        L: for<'a> Layout<'a, T>,
        Ax: Unsigned,
    {
        let mut shape = self.shape();

        let chunk_size_out = shape.iter().skip(Ax::USIZE + 1).product();
        let chunk_size_in = self.opt_chunk_size().min(chunk_size_out);

        let inner_loop_num = chunk_size_out / chunk_size_in;
        let mid_loop_num = shape.remove(Ax::USIZE);

        let mut out: Tensor<T, <S as Reduction<Ax>>::Output, Dynamic, Contiguous, P::Layout, P> =
            Tensor::alloc(shape);

        self.unchecked(
            chunk_size_in,
            chunk_size_out,
            mid_loop_num,
            inner_loop_num,
            &mut out,
        );
        out
    }
}
