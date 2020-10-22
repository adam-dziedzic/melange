//! `shape` contains all the tools to complement `typenum`
//! crate and efficiently use it for shape "arithmetics".
//!
//! As in `typenum`, this module contains two kinds of unsafe traits:
//! * type operators that act like type-level functions on type-level entities,
//! * marker traits that provide functions to interact with type-level entities at runtime.
//!
//! Type operators all share the Output associated
//! type that contains the type-level result of the operation the
//! trait represents.

use generic_array::ArrayLength;
use std::ops::*;
use typenum::operator_aliases::*;
use typenum::private::InternalMarker;
use typenum::type_operators::*;
use typenum::{ATerm, Bit, Equal, TArr, UInt, Unsigned, B0, B1, U0, U1};

pub fn intrinsic_strides_in_place(mut shape: Vec<usize>) -> Vec<usize> {
    let mut product = 1;
    for stride in shape.iter_mut().rev() {
        let tmp = product;
        product *= *stride;
        *stride = tmp;
    }

    shape
}

/// This trait "aliases" B1 (type-level bit one) for use in trait bounds.
/// It is especially useful with type-level binary operators.
///
/// # Example
///
/// ```
/// use melange::tensor::shape::{Same, TRUE};
/// 
/// fn test<S, Z>()
/// where
///     S: Same<Z>, // This bound is required by the following line
///     <S as Same<Z>>::Output: TRUE // Constrains "S Same Z" hold (Output = B1)
/// {}
/// ```
pub unsafe trait TRUE {}
unsafe impl TRUE for B1 {}

/// Zero-sized struct representing type-level dynamic dimension.
/// It implements comparisons with type level unsigned integers and
/// is considered equal to all of them. This involves Dyn is compatible
/// with any dimension.
pub struct Dyn;
impl<U> Cmp<U> for Dyn
where
    U: Dim,
{
    type Output = Equal;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &U) -> Self::Output {
        Equal
    }
}
impl<U, B> Cmp<Dyn> for UInt<U, B> {
    type Output = Equal;

    #[inline]
    fn compare<P: InternalMarker>(&self, _: &Dyn) -> Self::Output {
        Equal
    }
}

/// Marker trait implemented on type-level unsigned integers and `Dyn`
/// that provides a runtime equality check function.
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

/// Marker trait implemented on type-level unsigned integers only
/// that are the only valid type-level dimensions for a shape to be static.
pub unsafe trait StaticDim: Dim + Unsigned {}
unsafe impl<U, B> StaticDim for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
}

/// Marker trait implemented on `typenum`'s `TArr` containing a collection
/// of type-level unsigned integers or `Dyn`. Provides the `LEN` constant
/// that stores the number of axes in the shape and a `runtime_compat`
/// function.
pub unsafe trait Shape {
    /// Number of axes in the shape, i.e. order of the tensor.
    const LEN: usize;

    /// Checks the given slice `shape` against the type-level shape for compatibility.
    fn runtime_compat(shape: &[usize]) -> bool;
}

unsafe impl Shape for ATerm {
    const LEN: usize = 0;

    fn runtime_compat(shape: &[usize]) -> bool {
        shape.len() == 0
    }
}

unsafe impl<D, A> Shape for TArr<D, A>
where
    A: Shape,
    D: Dim,
{
    const LEN: usize = A::LEN + 1;

    fn runtime_compat(shape: &[usize]) -> bool {
        Self::LEN == shape.len()
            && D::runtime_eq(shape[A::LEN])
            && A::runtime_compat(&shape[..A::LEN])
    }
}

/// Marker trait implemented on shapes containing type-level
/// unsigned integers only. Provides means of conversion for
/// runtime use.
pub unsafe trait StaticShape: Shape {
    /// Number of elements in the tensor, i.e. product of all dimensions of the shape.
    const NUM_ELEMENTS: usize;
    /// Outputs a `Vec` containing the runtime version of the shape.
    fn to_vec() -> Vec<usize>;
    /// Outputs a `Vec` containing the intrinsic strides of the shape.
    /// Note that these strides do not account for the real layout.
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
        intrinsic_strides_in_place(Self::to_vec())
    }
}

/// Type operator that outputs the representative dimension of two
/// compatible dimensions:
/// * if both are the same type-level unsigned integers,
///   the representative is this integer,
/// * if both are `Dyn`, the representative is `Dyn`,
/// * if one is a type-level unsigned integer and the other is `Dyn`,
///   the representer is the integer.
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

/// Type operator that outputs the representative shape of two shapes
/// which is defined as the collection of the representatives of all
/// dimensions.
///
/// Note that this require both shape to have the same length and that
/// there is no guarantee on the output i.e. it can still be dynamic.
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

/// Type operator that outputs the representative shape of two shapes
/// which is defined as the collection of the representatives of all
/// dimensions.
///
/// This trait adds a further guarantee to `ReprShapeDyn` which is that
/// the output is guaranteed to be static. This means that the two shapes
/// must be coercible: they cannot both contain `Dyn` on the same axis.
///
/// Note that this require both shape to have the same length.
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
    TArr<<S as ReprDim<SRhs>>::Output, <A as ReprShape<T, ARhs>>::Output>:
        NumElements<T> + StaticShape,
{
    type Output = TArr<<S as ReprDim<SRhs>>::Output, <A as ReprShape<T, ARhs>>::Output>;
}

/// Binary type operator that outputs B1 if the implementor
/// dimension can be strided to Rhs.
pub unsafe trait StridedDim<Rhs> {
    type Output;
}

unsafe impl<U, B, V> StridedDim<V> for UInt<U, B>
where
    V: StaticDim,
    U: Div<V> + Rem<V>,
    <U as Rem<V>>::Output: IsGreater<U0>,
    <U as Div<V>>::Output: Add<Gr<<U as Rem<V>>::Output, U0>>,
{
    type Output = Sum<<U as Div<V>>::Output, Gr<<U as Rem<V>>::Output, U0>>;
}

unsafe impl<V> StridedDim<V> for Dyn {
    type Output = Dyn;
}

unsafe impl<U, B> StridedDim<Dyn> for UInt<U, B> {
    type Output = Dyn;
}

/// Binary type operator that outputs B1 if the implementor shape can be
/// strided to Rhs i.e. all dimensions can be strided to the dimension
/// on the respective axis of Rhs.
///
/// This trait adds a further guarantee to `StridedShapeDyn` which is that
/// the output is guaranteed to be static. This means that the two shapes
/// must be coercible: they cannot both contain `Dyn` on the same axis.
///
/// Note that this requires both shapes to have the same length.
pub unsafe trait StridedShape<Rhs> {
    type Output: StaticShape;
}

unsafe impl StridedShape<ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<S, A, SRhs, ARhs> StridedShape<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: StridedDim<SRhs>,
    A: StridedShape<ARhs>,
    TArr<<S as StridedDim<SRhs>>::Output, <A as StridedShape<ARhs>>::Output>: StaticShape,
{
    type Output = TArr<<S as StridedDim<SRhs>>::Output, <A as StridedShape<ARhs>>::Output>;
}

/// Binary type operator that outputs B1 if the implementor shape can be
/// strided to Rhs i.e. all dimensions can be strided to the dimension on
/// the respective axis of Rhs.
///
/// Note that this requires both shapes to have the same length. There is no
/// guarantee on the output i.e. it can still be dynamic.
pub unsafe trait StridedShapeDyn<Rhs> {
    type Output;
}

unsafe impl StridedShapeDyn<ATerm> for ATerm {
    type Output = ATerm;
}

unsafe impl<S, A, SRhs, ARhs> StridedShapeDyn<TArr<SRhs, ARhs>> for TArr<S, A>
where
    S: StridedDim<SRhs>,
    A: StridedShape<ARhs>,
{
    type Output = TArr<<S as StridedDim<SRhs>>::Output, <A as StridedShape<ARhs>>::Output>;
}

/// Binary trait operator that outputs B1 if the implementor shape
/// is compatible with Rhs i.e. all the dimensions on the respective
/// axes are compatible.
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

/// Binary trait operator that outputs B1 if the implementor shape
/// is fits in Rhs i.e. all the dimensions of the implementor are
/// less or equal to the dimanesions on the respective axes of Rhs.
///
/// Note than because `Dyn` is equal to all type-level unsigned integers,
/// it can fit in evrything and everything can fit in it.
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

/// Binary trait operator that outputs B1 if the implementor shape
/// can be broadcasted to Rhs. Broadcasting is valid if for all axes:
/// * dimensions are equal (`Dyn` is included but runtime check should be done)
/// * one of the dimensions is U1
///
/// Note that this requires both shapes to have the same length.
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
    Or<Or<Eq<S, SRhs>, Eq<S, U1>>, Eq<SRhs, U1>>: BitAnd<<A as Broadcast<ARhs>>::Output>,
{
    type Output = And<Or<Or<Eq<S, SRhs>, Eq<S, U1>>, Eq<SRhs, U1>>, <A as Broadcast<ARhs>>::Output>;
}

/// Marker trait implemented on static shapes that provides
/// a type-level number of elements and its runtime counterpart.
///
/// This is only useful for tensors stored on the stack that require
/// the number of elements to be known at compile time.
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

/// Binary trait operator that outputs B1 if the implementor shape
/// and Rhs have the same number of elements i.e. the products of their
/// dimensions are equal.
///
/// This is useful to perform compile-time reshape checks.
pub unsafe trait SameNumElements<T, Rhs> {
    type Output;
}

unsafe impl<T, S, A, Rhs> SameNumElements<T, Rhs> for TArr<S, A>
where
    Self: NumElements<T>,
    Rhs: NumElements<T>,
    <Self as NumElements<T>>::Output: IsEqual<<Rhs as NumElements<T>>::Output>,
{
    type Output = Eq<<Self as NumElements<T>>::Output, <Rhs as NumElements<T>>::Output>;
}

/// Conditionnal trait operator:
/// * outputs T if the implementor is B1,
/// * outputs Else otherwise.
pub trait If<T, Else> {
    type Output;
}

impl<T, Else> If<T, Else> for B1 {
    type Output = T;
}

impl<T, Else> If<T, Else> for B0 {
    type Output = Else;
}

/// Trait operator that replaces the dimension of the axis
/// having the (0-starting) index Ax (a type-level unsigned integer)
/// with U1.
pub trait Reduction<Ax> {
    type Output;
}

impl<Ax> Reduction<Ax> for ATerm {
    type Output = ATerm;
}

impl<Ax, D, Ar> Reduction<Ax> for TArr<D, Ar>
where
    Self: Len,
    Length<Self>: Sub<B1>,
    Ax: IsEqual<Sub1<Length<Self>>>,
    Ar: Reduction<Ax>,
    Eq<Ax, Sub1<Length<Self>>>: If<TArr<U1, Ar>, TArr<D, <Ar as Reduction<Ax>>::Output>>,
{
    type Output = <Eq<Ax, Sub1<Length<Self>>> as If<
        TArr<U1, Ar>,
        TArr<D, <Ar as Reduction<Ax>>::Output>,
    >>::Output;
}

/// Trait operator that computes the intrinsic optimal chunk size
/// i.e. the largest contiguous group of elements in storage
/// after a reduction performed on the axis at (0-starting)
/// index Ax (a type-level unsigned integer).
///
/// Note that this size is intrinsic to the shape and does not
/// take into account the real layout.
pub trait ReductionOptChunckSize<T, Ax>: StaticShape {
    type Output: Unsigned;
}

impl<Ax, T> ReductionOptChunckSize<T, Ax> for ATerm {
    type Output = U1;
}

impl<Ax, D, Ar, T> ReductionOptChunckSize<T, Ax> for TArr<D, Ar>
where
    D: Mul<<Ar as ReductionOptChunckSize<T, Ax>>::Output>,
    Self: Len + StaticShape,
    Length<Self>: Sub<B1>,
    Ax: IsEqual<Sub1<Length<Self>>>,
    Ar: ReductionOptChunckSize<T, Ax> + NumElements<T>,
    Eq<Ax, Sub1<Length<Self>>>: If<U1, Prod<D, <Ar as ReductionOptChunckSize<T, Ax>>::Output>>,
    <Eq<Ax, Sub1<Length<Self>>> as If<
        U1,
        Prod<D, <Ar as ReductionOptChunckSize<T, Ax>>::Output>,
    >>::Output: Unsigned,
{
    type Output = <Eq<Ax, Sub1<Length<Self>>> as If<
        U1,
        Prod<D, <Ar as ReductionOptChunckSize<T, Ax>>::Output>,
    >>::Output;
}

/// Type operator that outputs the dimension at (0-starting)
/// index Ax (a type-level unsigned integer) of the implementor shape.
pub trait At<Ax>: StaticShape {
    type Output: Unsigned;
}

impl<Ax> At<Ax> for ATerm {
    type Output = U1;
}

impl<Ax, D, Ar> At<Ax> for TArr<D, Ar>
where
    D: StaticDim,
    Self: Len,
    Length<Self>: Sub<B1>,
    Ax: IsEqual<Sub1<Length<Self>>>,
    Ar: At<Ax>,
    Eq<Ax, Sub1<Length<Self>>>: If<D, <Ar as At<Ax>>::Output>,
    <Eq<Ax, Sub1<Length<Self>>> as If<D, <Ar as At<Ax>>::Output>>::Output: Unsigned,
{
    type Output = <Eq<Ax, Sub1<Length<Self>>> as If<D, <Ar as At<Ax>>::Output>>::Output;
}

/// Type operator that inserts dimension S before the first axis.Abs
/// This is useful because dimensions are stored in reverse order in
/// the recursive `TArr` structure.
pub unsafe trait Insert<S> {
    type Output;
}

unsafe impl<S> Insert<S> for ATerm {
    type Output = TArr<S, ATerm>;
}

unsafe impl<S, A, Z> Insert<Z> for TArr<S, A>
where
    A: Insert<Z>,
{
    type Output = TArr<S, <A as Insert<Z>>::Output>;
}

/// Type operator that reverses the order of the axes in the implementor shape.
pub unsafe trait Transpose {
    type Output;
}

unsafe impl Transpose for ATerm {
    type Output = ATerm;
}

unsafe impl<S, A> Transpose for TArr<S, A>
where
    A: Transpose,
    <A as Transpose>::Output: Insert<S>,
{
    type Output = <<A as Transpose>::Output as Insert<S>>::Output;
}

/// 1D shape alias.
pub type Shape1D<S0> = TArr<S0, ATerm>;
/// 2D shape alias.
pub type Shape2D<S0, S1> = TArr<S1, TArr<S0, ATerm>>;
/// 3D shape alias.
pub type Shape3D<S0, S1, S2> = TArr<S2, TArr<S1, TArr<S0, ATerm>>>;
/// 4D shape alias.
pub type Shape4D<S0, S1, S2, S3> = TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>;
/// 5D shape alias.
pub type Shape5D<S0, S1, S2, S3, S4> = TArr<S4, TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>>;
/// 6D shape alias.
pub type Shape6D<S0, S1, S2, S3, S4, S5> =
    TArr<S5, TArr<S4, TArr<S3, TArr<S2, TArr<S1, TArr<S0, ATerm>>>>>>;
