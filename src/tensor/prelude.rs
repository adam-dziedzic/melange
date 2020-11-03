use super::allocation_policy::{DefaultPolicy, StackFirstPolicy};
use super::heap_layout::HeapLayout;
use super::slice_layout::SliceLayout;
use super::stack_layout::StackLayout;
use super::static_heap_layout::StaticHeapLayout;
use super::transpose_policy::{Contiguous, Strided};

pub use std::ops::*;
pub use super::layout::*;
pub use super::shape::*;
pub use super::tensor::Tensor;
pub use std::convert::TryFrom;
pub use super::core_ops::*;
pub use super::reduction::*;
pub use super::linear_algebra::*;

/// Default static tensor stored on the heap.
pub type StaticTensor<T, S> = Tensor<T, S, Static, Contiguous, StaticHeapLayout<T, S>, DefaultPolicy>;

/// Default dynamic tensor stored on the heap.
pub type DynamicTensor<T, S> = Tensor<T, S, Dynamic, Contiguous, HeapLayout<T>, DefaultPolicy>;

/// Default static view tensor.
pub type StaticSliceTensor<'a, T, S> = Tensor<T, S, Static, Contiguous, SliceLayout<'a, T>, DefaultPolicy>;

/// Default dynamic view tensor.
pub type DynamicSliceTensor<'a, T, S> = Tensor<T, S, Dynamic, Contiguous, SliceLayout<'a, T>, DefaultPolicy>;

/// Default strided static view tensor.
pub type StridedStaticSliceTensor<'a, T, S> = Tensor<T, S, Static, Strided, SliceLayout<'a, T>, DefaultPolicy>;

/// Default strided dynamic view tensor.
pub type StridedDynamicSliceTensor<'a, T, S> = Tensor<T, S, Dynamic, Strided, SliceLayout<'a, T>, DefaultPolicy>;

/// Stack allocated tensor.
pub type StackTensor<T, S> = Tensor<T, S, Static, Contiguous, StackLayout<T, S>, StackFirstPolicy>;

/// Default view with stack allocation policy.
pub type StackSliceTensor<'a, T, S> =
    Tensor<T, S, Static, Contiguous, SliceLayout<'a, T>, StackFirstPolicy>;
