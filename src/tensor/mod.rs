//! `tensor` is a collection of tools to interact with multidimensional arrays
//! that are at the core of ML pipelines. It defines various ways to store data
//! and optimized mathematical operations. Contrary to other ndarray modules
//! such as numpy in Python or ndarray in Rust, this module allows full size
//! checking at compile time thanks to type level integers from the `typenum`
//! crate.

pub mod allocation_policy;
pub mod core_ops;
pub mod heap_layout;
pub mod layout;
pub mod linear_algebra;
pub mod prelude;
pub mod reduction;
pub mod shape;
pub mod slice_layout;
pub mod stack_layout;
pub mod static_heap_layout;
pub mod tensor;
pub mod transpose_policy;
