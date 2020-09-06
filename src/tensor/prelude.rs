use super::static_heap_layout::StaticHeapLayout;
use super::heap_layout::HeapLayout;
use super::stack_layout::StackLayout;
use super::slice_layout::SliceLayout;
use super::allocation_policy::{DefaultPolicy, StackFirstPolicy};

pub use super::tensor::Tensor;

pub type StaticTensor<T, S> = Tensor<T, S, StaticHeapLayout<T, S>, DefaultPolicy>;
pub type DynamicTensor<T, S> = Tensor<T, S, HeapLayout<T>, DefaultPolicy>;
pub type SliceTensor<'a, T, S> = Tensor<T, S, SliceLayout<'a, T>, DefaultPolicy>;
pub type StackTensor<T, S> = Tensor<T, S, StackLayout<T, S>, StackFirstPolicy>;
pub type StackSliceTensor<'a, T, S> = Tensor<T, S, SliceLayout<'a, T>, StackFirstPolicy>;
