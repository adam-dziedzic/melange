extern crate cblas;
use cblas::Transpose;

#[derive(Debug, PartialEq)]
pub struct Contiguous;

#[derive(Debug, PartialEq)]
pub struct Transposed;

#[derive(Debug, PartialEq)]
pub struct Strided;

pub trait TransposePolicy {
    type Transposed;
}

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
