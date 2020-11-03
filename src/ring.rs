//! `ring` defines the `Ring` trait that provides useful
//! features of algebraic rings such as identity elements
//! and multiplicative inversion.
//! 
//! It is implemented for all numeric primitive types
//! thanks to `expand_operations` procedural macro from
//! `melange_macros` crate.

use melange_macros::expand_impl;
use std::ops::*;

pub trait Ring {
    const ZERO: Self;
    const ONE: Self;

    fn inv(self) -> Self
    where
        Self: Sized + Div<Output = Self>,
    {
        Self::ONE / self
    }
}

#[expand_impl(
    _f64<T=f64>,
    _f32<T=f32>,
)]
impl<T> Ring for T {
    const ZERO: T = 0.0;
    const ONE: T = 1.0;
}

#[expand_impl(
    _u128<T=u128>,
    _u64<T=u64>,
    _u32<T=u32>,
    _u16<T=u16>,
    _u8<T=u8>,
    _i128<T=i128>,
    _i64<T=i64>,
    _i32<T=i32>,
    _i16<T=i16>,
    _i8<T=i8>,
)]
impl<T> Ring for T {
    const ZERO: T = 0;
    const ONE: T = 1;
}
