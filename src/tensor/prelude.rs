use super::stack_layout::StackLayout;
use super::slice_layout::SliceLayout;

pub use super::tensor::Tensor;

pub type StackTensor<T, S> = Tensor<T, S, StackLayout<T, S>>;
pub type SliceTensor<'a, T, S> = Tensor<T, S, SliceLayout<'a, T>>;