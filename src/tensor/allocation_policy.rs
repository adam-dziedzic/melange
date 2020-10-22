//! `allocation_policy` defines the traits `StaticAllocationPolicy` and
//! `DynamicAllocationPolicy` that both contain an associated `Layout` type
//! that should be used by operations when allocating. When an operation
//! involves more than one tensor, the policy of `self` should be applied and
//! passed to the new tensor. Policies are simply zero-sized structs that
//! implement both traits.

use super::heap_layout::HeapLayout;
use super::layout::{Alloc, LayoutMut};
use super::shape::{NumElements, StaticShape};
use super::stack_layout::StackLayout;
use super::static_heap_layout::StaticHeapLayout;

/// Trait that defines the `Layout` that should be used with the implementor
/// policy in the context of statically sized (compile time) tensors.
pub trait StaticAllocationPolicy<T, S> {
    type Layout: Default + for<'a> LayoutMut<'a, T>;
}

/// Trait that defines the `Layout` that should be used with the implementor
/// policy in the context of dynamically sized (run time) tensors.
pub trait DynamicAllocationPolicy<T> {
    type Layout: Alloc + for<'a> LayoutMut<'a, T>;
}

/// This policy uses `StaticHeapLayout` for statically sized tensors
/// and `HeapLayout` for dynamically sized tensors.
#[derive(Debug, PartialEq, Clone)]
pub struct DefaultPolicy;

/// This policy uses `StackLayout` for statically sized tensors
/// and `HeapLayout` for dynamically sized tensors.
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
