//! `transpose_policy` defines the trait `BLASPolicy` that dictates the behavior
//! of BLAS operations by passing `BLAS_TRANSPOSE` associated constant.
//! The trait `TransposePolicy` defines the relationship between transpose policies:
//! the associated type `Transposed` corresponds to the policy of the transposed tensor.
//!
//! Only transposition, broadcasting and striding affect the transpose policy.
//!
//! Policies are simply zero-sized structs that implement `TransposePolicy`.
//! If they allow BLAS operations, they also implement `BLASPolicy`.

extern crate cblas;
use cblas::Transpose;

/// Policy used with contiguous tensors directly compatible with BLAS.
#[derive(Debug, PartialEq, Clone)]
pub struct Contiguous;

/// Policy used with contiguous tensors transposed by reverting the axes order
/// whose layout remains untransposed from BLAS point of view.  
#[derive(Debug, PartialEq, Clone)]
pub struct Transposed;

/// Policy used with non contiguous tensors that are not suitable
/// for BLAS operations.
#[derive(Debug, PartialEq, Clone)]
pub struct Strided;

/// Defines the policy of the corresponding transposed tensor.
pub trait TransposePolicy {
    type Transposed;
}

/// Defines the constant that should be passed to BLAS operation.
/// Should only be implemented for contiguous policies.
pub trait BLASPolicy {
    const BLAS_TRANSPOSE: Transpose;
}

impl TransposePolicy for Contiguous {
    type Transposed = Transposed;
}

impl BLASPolicy for Contiguous {
    const BLAS_TRANSPOSE: Transpose = Transpose::None;
}

impl TransposePolicy for Transposed {
    type Transposed = Contiguous;
}

impl BLASPolicy for Transposed {
    const BLAS_TRANSPOSE: Transpose = Transpose::Ordinary;
}

impl TransposePolicy for Strided {
    type Transposed = Strided;
}
