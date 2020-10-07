use super::allocation_policy::{DefaultPolicy, StackFirstPolicy};
use super::heap_layout::HeapLayout;
use super::slice_layout::SliceLayout;
use super::stack_layout::StackLayout;
use super::static_heap_layout::StaticHeapLayout;
use super::transpose_policy::{Contiguous, Strided, Transposed};

pub use super::layout::*;
pub use super::shape::*;
pub use super::tensor::Tensor;

pub type StaticTensor<T, S> = Tensor<T, S, Contiguous, StaticHeapLayout<T, S>, DefaultPolicy>;
pub type DynamicTensor<T, S> = Tensor<T, S, Contiguous, HeapLayout<T>, DefaultPolicy>;
pub type SliceTensor<'a, T, S> = Tensor<T, S, Contiguous, SliceLayout<'a, T>, DefaultPolicy>;
pub type TransposedSliceTensor<'a, T, S> =
    Tensor<T, S, Transposed, SliceLayout<'a, T>, DefaultPolicy>;
pub type StridedSliceTensor<'a, T, S> = Tensor<T, S, Strided, SliceLayout<'a, T>, DefaultPolicy>;
pub type StackTensor<T, S> = Tensor<T, S, Contiguous, StackLayout<T, S>, StackFirstPolicy>;
pub type ContiguousStackSliceTensor<'a, T, S> =
    Tensor<T, S, Contiguous, SliceLayout<'a, T>, StackFirstPolicy>;
pub type TransposedStackSliceTensor<'a, T, S> =
    Tensor<T, S, Transposed, SliceLayout<'a, T>, StackFirstPolicy>;
pub type StridedStackSliceTensor<'a, T, S> =
    Tensor<T, S, Strided, SliceLayout<'a, T>, StackFirstPolicy>;
