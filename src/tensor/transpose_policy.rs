extern crate cblas;
use cblas::Transpose;

#[derive(Debug, PartialEq)]
pub struct Contiguous;

#[derive(Debug, PartialEq)]
pub struct Transposed;

#[derive(Debug, PartialEq)]
pub struct Strided;

pub trait TransposePolicy {
    const BLAS_TRANSPOSE: Transpose;
}

impl TransposePolicy for Contiguous {
    const BLAS_TRANSPOSE: Transpose = Transpose::None;
}

impl TransposePolicy for Transposed {
    const BLAS_TRANSPOSE: Transpose = Transpose::Ordinary;
}
