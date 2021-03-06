use crate::tensor::allocation_policy::{DynamicAllocationPolicy, StaticAllocationPolicy};
use crate::tensor::prelude::*;
use crate::tensor::transpose_policy::Contiguous;
use std::cell::RefCell;
use std::fmt;
use std::ops::{AddAssign, Deref};
use std::rc::Rc;

/// Groups the value, gradient option and backpropagation closure as a sigle entity.
/// Fields are public in the parent module (`backprop`).
pub struct BackpropNode<T, S, C, L, P, Lgrad, Cback, Lback, Pback> {
    pub(super) value: Tensor<T, S, C, L, P>,
    pub(super) grad: Option<Tensor<T, S, Contiguous, Lgrad, P>>,
    pub(super) backward_op_name: &'static str,
    pub(super) backward_closure: Box<dyn Fn(Tensor<T, S, Cback, Lback, Pback>) -> ()>,
}

impl<T, S, C, L, P, Lgrad, Cback, Lback, Pback> fmt::Debug
    for BackpropNode<T, S, C, L, P, Lgrad, Cback, Lback, Pback>
where
    T: fmt::Debug,
    S: fmt::Debug,
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
pub struct Variable<T, S, C, L, P, Lgrad, Cback, Lback, Pback>(
    pub(super) Rc<RefCell<BackpropNode<T, S, C, L, P, Lgrad, Cback, Lback, Pback>>>,
);

impl<T, S, C, L, P, Lgrad, Cback, Lback, Pback> fmt::Debug
    for Variable<T, S, C, L, P, Lgrad, Cback, Lback, Pback>
where
    T: fmt::Debug,
    S: fmt::Debug,
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

impl<T, S, C, L, P, Lgrad, Cback, Lback, Pback> Deref
    for Variable<T, S, C, L, P, Lgrad, Cback, Lback, Pback>
{
    type Target = Rc<RefCell<BackpropNode<T, S, C, L, P, Lgrad, Cback, Lback, Pback>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, S, C, L, P, Lgrad, Cback, Lback, Pback> Variable<T, S, C, L, P, Lgrad, Cback, Lback, Pback>
where
    Tensor<T, S, Contiguous, Lgrad, P>: Clone,
{
    // Returns an option to a copy of the gradient.
    pub fn grad(&self) -> Option<Tensor<T, S, Contiguous, Lgrad, P>> {
        self.borrow().grad.clone()
    }
}

impl<T, S, C, L, P, Cback, Lback, Pback> Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
where
    S: StaticShape,
    L: for<'a> Layout<'a, T>,
    P: StaticAllocationPolicy<T, S>,
{
    /// Add given gradient to the retained gradient if needed and
    /// backpropagate using the closure.
    ///
    /// To initiate backpropagation a tensor full of ones is a good choice.
    pub fn backward(&self, grad: Tensor<T, S, Cback, Lback, Pback>)
    where
        T: Send + Sync + Copy + AddAssign,
        Lback: for<'a> Layout<'a, T>,
    {
        {
            let mut node = self.borrow_mut();
            if let Some(current_grad) = &mut node.grad {
                current_grad.add_(&grad);
            }
        }

        let node = self.borrow();
        (node.backward_closure)(grad);
    }

    /// Create a new variable that retains its gradient if require_grad is true
    /// by moving the given tensor.
    pub fn new(tensor: Tensor<T, S, C, L, P>, require_grad: bool) -> Self {
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

impl<T, S, C, L, P, Cback, Lback, Pback> Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
where
    L: for<'a> Layout<'a, T>,
    P: DynamicAllocationPolicy<T>,
{
    /// Dynamic version of the backpropagation function.
    /// Name has to be different because specialization is not yet
    /// available in stable Rust.
    ///
    /// Add given gradient to the retained gradient if needed and
    /// backpropagate using the closure.
    ///
    /// To initiate backpropagation a tensor full of ones is a good choice.
    pub fn backward_dynamic(&self, grad: Tensor<T, S, Cback, Lback, Pback>)
    where
        T: Send + Sync + Copy + AddAssign,
        P: DynamicAllocationPolicy<T>,
        P::Layout: for<'a> Layout<'a, T>,
        Lback: for<'a> Layout<'a, T>,
    {
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
                current_grad.add_dynamic_(&grad);
            }
        }

        let node = self.borrow();
        (node.backward_closure)(grad);
    }
}

impl<T, S, C, L, P, Lgrad, Cback, Lback, Pback> Clone
    for Variable<T, S, C, L, P, Lgrad, Cback, Lback, Pback>
{
    fn clone(&self) -> Self {
        Variable(Rc::clone(&self.0))
    }
}
