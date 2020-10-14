use road_ai_macros::expand_operations;

pub trait Ring<T> {
    const ZERO: T;
    const ONE: T;
}

#[expand_operations(
    _f64<T=f64>,
    _f32<T=f32>,
)]
impl<T> Ring<T> for T {
    const ZERO: T = 0.0;
    const ONE: T = 1.0;
}

#[expand_operations(
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
impl<T> Ring<T> for T {
    const ZERO: T = 0;
    const ONE: T = 1;
}
