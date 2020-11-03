use crate::tensor::allocation_policy::{DynamicAllocationPolicy, StaticAllocationPolicy};
use crate::tensor::prelude::*;
use crate::tensor::transpose_policy::Contiguous;
use std::cell::RefCell;
use std::fmt;
use std::ops::{AddAssign, Deref};
use std::rc::Rc;
use melange_macros::expand_impl;

/// Create a new variable that retains its gradient if require_grad is true
/// by moving the given tensor.
pub trait New<T> {
    fn new(tensor: T, require_grad: bool) -> Self;
}

/// Groups the value, gradient option and backpropagation closure as a sigle entity.
/// Fields are public in the parent module (`backprop`).
pub struct BackpropNode<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad> {
    pub(super) value: Tensor<T, S, D, C, L, P>,
    pub(super) grad: Option<Tensor<T, S, D, Contiguous, Lgrad, P>>,
    pub(super) backward_op_name: &'static str,
    pub(super) backward_closure: Box<dyn Fn(Tensor<T, Sback, Dback, Cback, Lback, Pback>) -> ()>,
}

impl<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad> fmt::Debug
    for BackpropNode<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>
where
    T: fmt::Debug,
    S: fmt::Debug,
    D: fmt::Debug,
    C: fmt::Debug,
    L: fmt::Debug,
    P: fmt::Debug,
    Lgrad: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("value", &self.value)
            .field("grad", &self.grad)
            .field("backward_op", &self.backward_op_name)
            .finish()
    }
}

/// Core type of `backprop` module that represents a node in the computation
/// graph. It contains a combination of `Rc` and `RefCell` to allow
/// mutable reference counting of the actual `BackpropNode`s.
pub struct Variable<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>(
    pub(super) Rc<RefCell<BackpropNode<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>>>,
);

impl<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad> fmt::Debug
    for Variable<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>
where
    T: fmt::Debug,
    S: fmt::Debug,
    D: fmt::Debug,
    C: fmt::Debug,
    L: fmt::Debug,
    P: fmt::Debug,
    Lgrad: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.borrow();
        node.fmt(f)
    }
}

impl<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad> Deref
    for Variable<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>
{
    type Target = Rc<RefCell<BackpropNode<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad> Variable<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>
where
    Tensor<T, S, D, Contiguous, Lgrad, P>: Clone,
{
    // Returns an option to a copy of the gradient.
    pub fn grad(&self) -> Option<Tensor<T, S, D, Contiguous, Lgrad, P>> {
        self.borrow().grad.clone()
    }
}

impl<T, S, C, L, P, Cback, Lback, Pback, Lgrad> Variable<T, S, Static, C, L, P, S, Static, Cback, Lback, Pback, Lgrad>
where
    T: Send + Sync + Copy + AddAssign,
    Lgrad: for<'a> LayoutMut<'a, T> + for<'a> Layout<'a, T>,
    Lback: for<'a> Layout<'a, T>,
{
    /// Add given gradient to the retained gradient if needed and
    /// backpropagate using the closure.
    ///
    /// To initiate backpropagation a tensor full of ones is a good choice.
    pub fn backward(&self, grad: Tensor<T, S, Static, Cback, Lback, Pback>) {
        {
            let mut node = self.borrow_mut();
            if let Some(current_grad) = &mut node.grad {
                current_grad.add_(&grad);
            }
        }

        let node = self.borrow();
        (node.backward_closure)(grad);
    }
}

#[expand_impl(
    backward<D=Static, Dback=Dynamic>,
    backward<D=Dynamic, Dback=Static>,
    backward<D=Dynamic, Dback=Dynamic>,
)]
impl<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad> Variable<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>
where
    T: Send + Sync + Copy + AddAssign,
    S: Same<Sback>,
    <S as Same<Sback>>::Output: TRUE,
    L: for<'a> Layout<'a, T>,
    Lgrad: for<'a> LayoutMut<'a, T> + for<'a> Layout<'a, T>,
    Lback: for<'a> Layout<'a, T>,
{
    /// Add given gradient to the retained gradient if needed and
    /// backpropagate using the closure.
    ///
    /// To initiate backpropagation a tensor full of ones is a good choice.
    pub fn operation(&self, grad: Tensor<T, Sback, Dback, Cback, Lback, Pback>) {
        {
            let node = self.borrow();
            assert_eq!(
                node.value.shape(),
                grad.shape(),
                "Variable value and backward gradient must have the same shape. Got {:?} and {:?}.",
                node.value.shape(),
                grad.shape()
            );
        }

        {
            let mut node = self.borrow_mut();
            if let Some(current_grad) = &mut node.grad {
                current_grad.add_(&grad);
            }
        }

        let node = self.borrow();
        (node.backward_closure)(grad);
    }
}

impl<T, S, C, L, P, Sback, Dback, Cback, Lback, Pback> New<Tensor<T, S, Static, C, L, P>> for Variable<T, S, Static, C, L, P, Sback, Dback, Cback, Lback, Pback, P::Layout>
where
    P: StaticAllocationPolicy<T, S>,
{
    fn new(tensor: Tensor<T, S, Static, C, L, P>, require_grad: bool) -> Self {
        Variable(Rc::new(RefCell::new(BackpropNode {
            value: tensor,
            grad: if require_grad {
                Some(Tensor::default())
            } else {
                None
            },
            backward_op_name: "no_op",
            backward_closure: Box::new(|_grad| ()),
        })))
    }
}

impl<T, S, C, L, P, Sback, Dback, Cback, Lback, Pback> New<Tensor<T, S, Dynamic, C, L, P>> for Variable<T, S, Dynamic, C, L, P, Sback, Dback, Cback, Lback, Pback, P::Layout>
where
    P: DynamicAllocationPolicy<T>,
    L: for<'a> Layout<'a, T>,
{
    fn new(tensor: Tensor<T, S, Dynamic, C, L, P>, require_grad: bool) -> Self {
        let shape = tensor.shape();
        Variable(Rc::new(RefCell::new(BackpropNode {
            value: tensor,
            grad: if require_grad {
                Some(Tensor::alloc(shape))
            } else {
                None
            },
            backward_op_name: "no_op",
            backward_closure: Box::new(|_grad| ()),
        })))
    }
}

impl<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad> Clone
    for Variable<T, S, D, C, L, P, Sback, Dback, Cback, Lback, Pback, Lgrad>
{
    fn clone(&self) -> Self {
        Variable(Rc::clone(&self.0))
    }
}
