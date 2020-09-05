use typenum::{Bit, UInt, U1, B1, TArr, ATerm, Equal, Unsigned};
use typenum::type_operators::*;
use typenum::operator_aliases::*;
use typenum::private::InternalMarker;
use generic_array::ArrayLength;
// use typenum::marker_traits::*;
use std::ops::*;

pub unsafe trait TRUE {}
unsafe impl TRUE for B1 {}

pub struct Dyn;
impl<U> Cmp<U> for Dyn
where
    U: Dim
{
    type Output = Equal;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &U) -> Self::Output {
        Equal
    }
}
impl<U, B> Cmp<Dyn> for UInt<U, B>
{
    type Output = Equal;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &Dyn) -> Self::Output {
        Equal
    }
}

pub unsafe trait Dim {
    fn runtime_eq(dim: usize) -> bool;
}

unsafe impl<U, B> Dim for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
    fn runtime_eq(dim: usize) -> bool {
        Self::to_usize() == dim
    }
}

unsafe impl Dim for Dyn {
    fn runtime_eq(_dim: usize) -> bool {
        true
    }
}

pub unsafe trait StaticDim: Dim + Unsigned {}
unsafe impl<U, B> StaticDim for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{}

pub unsafe trait Shape {
    const LEN: usize;

    fn runtime_compat(shape: &[usize]) -> bool;
}

unsafe impl Shape for ATerm {
    const LEN: usize = 0;

    fn runtime_compat(shape: &[usize]) -> bool {
        shape.len() == 0
    }
}

// unsafe impl<D> Shape for TArr<D, ATerm>
// where
//     D: Dim,
// {
//     const LEN: usize = 1;

//     fn runtime_compat(shape: &[usize]) -> bool {
//         shape.len() == 1 &&
//         D::runtime_eq(shape[0])
//     }
// }

unsafe impl<D, A> Shape for TArr<D, A>
where
    A: Shape,
    D: Dim,
{
    const LEN: usize = A::LEN + 1;

    fn runtime_compat(shape: &[usize]) -> bool {
        Self::LEN == shape.len() &&
        D::runtime_eq(shape[A::LEN]) &&
        A::runtime_compat(&shape[..A::LEN])
    }
}

pub unsafe trait StaticShape: Shape {
    const NUM_ELEMENTS: usize;
    
    fn to_vec() -> Vec<usize>;
    fn strides() -> Vec<usize>;
}

unsafe impl StaticShape for ATerm {
    const NUM_ELEMENTS: usize = 1;

    #[inline]
    fn to_vec() -> Vec<usize> {
        Vec::new()
    }

    #[inline]
    fn strides() -> Vec<usize> {
        Vec::new()
    }
}

// unsafe impl<D> StaticShape for TArr<D, ATerm>
// where
//     D: StaticDim,
// {
//     const NUM_ELEMENTS: usize = D::USIZE;

//     #[inline]
//     fn to_vec() -> Vec<usize> {
//         vec![D::USIZE]
//     }

//     #[inline]
//     fn strides() -> Vec<usize> {
//         vec![1]
//     }
// }

unsafe impl<D, A> StaticShape for TArr<D, A>
where
    A: StaticShape,
    D: StaticDim,
{
    const NUM_ELEMENTS: usize = D::USIZE * A::NUM_ELEMENTS;

    #[inline]
    fn to_vec() -> Vec<usize> {
        let mut vec = A::to_vec();
        vec.push(D::USIZE);

        vec
    }

    #[inline]
    fn strides() -> Vec<usize> {
        let mut product = 1;
        let mut strides_vec = Self::to_vec();

        for stride in strides_vec.iter_mut().rev() {
            let tmp = product;
            product *= *stride;
            *stride = tmp;
        }

        strides_vec
    }
}

pub unsafe trait ReprDim<Rhs>: IsEqual<Rhs> {
    type Output;
}

unsafe impl<U, B, V> ReprDim<V> for UInt<U, B>
where
    Self: IsEqual<V>,
{
    type Output = Self;
}

unsafe impl<V> ReprDim<V> for Dyn
where
    Self: IsEqual<V>,
{
    type Output = V;
}

pub unsafe trait ReprShapeDyn<T, Rhs>: Same<Rhs> {
    type Output;
}

unsafe impl<T> ReprShapeDyn<T, ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<T, S, A, SRhs, ARhs> ReprShapeDyn<T, TArr<SRhs, ARhs>> for TArr<S, A>
where
    Self: Same<TArr<SRhs, ARhs>>,
    S: ReprDim<SRhs>,
    A: ReprShapeDyn<T, ARhs>,
{
    type Output = TArr<<S as ReprDim<SRhs>>::Output, <A as ReprShapeDyn<T, ARhs>>::Output>;
}

pub unsafe trait ReprShape<T, Rhs>: Same<Rhs> {
    type Output: NumElements<T> + StaticShape;
}

unsafe impl<T> ReprShape<T, ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<T, S, A, SRhs, ARhs> ReprShape<T, TArr<SRhs, ARhs>> for TArr<S, A>
where
    Self: Same<TArr<SRhs, ARhs>>,
    S: ReprDim<SRhs>,
    A: ReprShape<T, ARhs>,
    TArr<<S as ReprDim<SRhs>>::Output, <A as ReprShape<T, ARhs>>::Output>: NumElements<T> + StaticShape,
{
    type Output = TArr<<S as ReprDim<SRhs>>::Output, <A as ReprShape<T, ARhs>>::Output>;
}

pub unsafe trait Same<Rhs> {
    type Output;
}

unsafe impl Same<ATerm> for ATerm {
    type Output = B1;
}

unsafe impl<S, A, SRhs, ARhs> Same<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: IsEqual<SRhs>,
    A: Same<ARhs>,
    Eq<S, SRhs>: BitAnd<<A as Same<ARhs>>::Output>,
{
    type Output = And<Eq<S, SRhs>, <A as Same<ARhs>>::Output>;
}

pub unsafe trait FitIn<Rhs> {
    type Output;
}

unsafe impl FitIn<ATerm> for ATerm {
    type Output = B1;
}

unsafe impl<S, A> FitIn<TArr<S, A>> for ATerm {
    type Output = B1;
}

unsafe impl<S, A, SRhs, ARhs> FitIn<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: IsLessOrEqual<SRhs>,
    A: FitIn<ARhs>,
    LeEq<S, SRhs>: BitAnd<<A as FitIn<ARhs>>::Output>,
{
    type Output = And<LeEq<S, SRhs>, <A as FitIn<ARhs>>::Output>;
}

pub unsafe trait Broadcast<Rhs> {
    type Output;
}

unsafe impl Broadcast<ATerm> for ATerm {
    type Output = B1;
}

unsafe impl<S, A> Broadcast<TArr<S, A>> for ATerm {
    type Output = B1;
}

unsafe impl<S, A, SRhs, ARhs> Broadcast<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: IsEqual<SRhs> + IsEqual<U1>,
    SRhs: IsEqual<U1>,
    Eq<S, SRhs>: BitOr<Eq<S, U1>>,
    Or<Eq<S, SRhs>, Eq<S, U1>>: BitOr<Eq<SRhs, U1>>,
    A: Broadcast<ARhs>,
    Or<Or<Eq<S, SRhs>, Eq<S, U1>>, Eq<SRhs, U1>>: BitAnd<<A as Broadcast<ARhs>>::Output>
{
    type Output = And<Or<Or<Eq<S, SRhs>, Eq<S, U1>>, Eq<SRhs, U1>>, <A as Broadcast<ARhs>>::Output>;
}

pub unsafe trait NumElements<T> {
    type Output: Unsigned + ArrayLength<T>;

    fn num_elements() -> usize {
        <Self::Output as Unsigned>::to_usize()
    }
}

unsafe impl<T> NumElements<T> for ATerm {
    type Output = U1;
}

unsafe impl<T, S, A> NumElements<T> for TArr<S, A>
where
    A: NumElements<T>,
    S: StaticDim + Mul<<A as NumElements<T>>::Output>,
    Prod<S, <A as NumElements<T>>::Output>: Unsigned + ArrayLength<T>,
{
    type Output = Prod<S, <A as NumElements<T>>::Output>;
}

pub unsafe trait SameNumElements<T, Rhs> {
    type Output;
}

unsafe impl<T, S, A, Rhs> SameNumElements<T, Rhs> for TArr<S, A>
where
    Self: NumElements<T>,
    Rhs: NumElements<T>,
    <Self as NumElements<T>>::Output: IsEqual<<Rhs as NumElements<T>>::Output>
{
    type Output = Eq<<Self as NumElements<T>>::Output, <Rhs as NumElements<T>>::Output>;
}

pub type Shape1D<S0> = TArr<S0, ATerm>;
pub type Shape2D<S0, S1> = TArr<S1, TArr<S0, ATerm>>;
pub type Shape3D<S0, S1, S2> = TArr<S2, TArr<S1, TArr<S0, ATerm>>>;
pub type Shape4D<S0, S1, S2, S3> = TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>;
pub type Shape5D<S0, S1, S2, S3, S4> = TArr<S4, TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>>;
pub type Shape6D<S0, S1, S2, S3, S4, S5> = TArr<S5, TArr<S4, TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>>>;