use super::static_heap_layout::StaticHeapLayout;
use super::heap_layout::HeapLayout;
use super::stack_layout::StackLayout;
use super::layout::{LayoutMut, Alloc};
use super::shape::{StaticShape, NumElements};

pub trait StaticAllocationPolicy<T, S> {
    type Layout: Default + for<'a> LayoutMut<'a, T>;
}

pub trait DynamicAllocationPolicy<T> {
    type Layout: Alloc + for<'a> LayoutMut<'a, T>;
}

#[derive(Debug, PartialEq)]
pub struct DefaultPolicy;

#[derive(Debug, PartialEq)]
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
{
    type Layout = StackLayout<T, S>;
}

impl<T> DynamicAllocationPolicy<T> for StackFirstPolicy
where
    T: Default + Clone + 'static,
{
    type Layout = HeapLayout<T>;
}
