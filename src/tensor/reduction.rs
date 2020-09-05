use std::ops::*;
use rayon::prelude::*;
use typenum::Unsigned;
use super::tensor::Tensor;
use super::shape::{Reduction, ReductionOptChunckSize, At};
use super::layout::{Layout, LayoutMut, OpsDefaultOutput, OpsAllocOutput};

impl<T, S, L> Tensor<T, S, L>
where
    S: std::fmt::Debug,
    L: for<'a> Layout<'a, T> + std::fmt::Debug,
    T: Send + Sync + Copy + std::fmt::Debug,
{
    pub fn sum<Ax>(&self) -> Tensor<T, <S as Reduction<Ax>>::Output, L::Default>
    where
        T: AddAssign,
        S: Reduction<Ax> + ReductionOptChunckSize<T, Ax> + At<Ax>,
        L: OpsDefaultOutput<T, <S as Reduction<Ax>>::Output>,
    {
        let chunk_size_out = <S as ReductionOptChunckSize<T, Ax>>::Output::USIZE;
        let chunk_size_in = self.opt_chunk_size().min(chunk_size_out);

        let inner_loop_num = chunk_size_out / chunk_size_in;
        let mid_loop_num = <S as At<Ax>>::Output::USIZE;

        let mut in_iter = self.chunks(chunk_size_in);

        let mut out: Tensor<T, <S as Reduction<Ax>>::Output, L::Default> = Tensor::default();
    
        for chunk_o in out.chunks_mut(chunk_size_out) {
            for _ in 0..mid_loop_num {
                for j in 0..inner_loop_num {
                    let chunk_i = in_iter.next().unwrap();
    
                    chunk_o
                        .par_iter_mut()
                        .skip(j * chunk_size_in)
                        .zip(chunk_i.par_iter())
                        .for_each(|(o, i)| *o += *i);
                }
            }
        }
    
        out
    }
}