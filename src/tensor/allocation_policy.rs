use super::heap_layout::HeapLayout;
use super::layout::{Alloc, LayoutMut};
use super::shape::{NumElements, StaticShape};
use super::stack_layout::StackLayout;
use super::static_heap_layout::StaticHeapLayout;

pub trait StaticAllocationPolicy<T, S> {
    type Layout: Default + for<'a> LayoutMut<'a, T>;
}

pub trait DynamicAllocationPolicy<T> {
    type Layout: Alloc + for<'a> LayoutMut<'a, T>;
}

#[derive(Debug, PartialEq, Clone)]
pub struct DefaultPolicy;

#[derive(Debug, PartialEq, Clone)]
pub struct StackFirstPolicy;

impl<T, S> StaticAllocationPolicy<T, S> for DefaultPolicy
where
    T: Default + Clone + 'static,
    S: StaticShape,
{
    type Layout = StaticHeapLayout<T, S>;
}

impl<T> DynamicAllocationPolicy<T> for DefaultPolicy
where
    T: Default + Clone + 'static,
{
    type Layout = HeapLayout<T>;
}

impl<T, S> StaticAllocationPolicy<T, S> for StackFirstPolicy
where
    T: Default + 'static,
    S: StaticShape + NumElements<T>,
    <S as NumElements<T>>::Output: 'static,
{
    type Layout = StackLayout<T, S>;
}

impl<T> DynamicAllocationPolicy<T> for StackFirstPolicy
where
    T: Default + Clone + 'static,
{
    type Layout = HeapLayout<T>;
}
