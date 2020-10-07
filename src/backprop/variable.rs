use crate::tensor::prelude::*;
use crate::tensor::allocation_policy::{DynamicAllocationPolicy, StaticAllocationPolicy};
use crate::tensor::transpose_policy::Contiguous;
use std::cell::RefCell;
use std::ops::{AddAssign, Deref};
use std::rc::Rc;

pub struct BackpropNode<T, S, C, L, P, Lgrad>
{
    pub(super) value: Tensor<T, S, C, L, P>,
    pub(super) grad: Option<Tensor<T, S, Contiguous, Lgrad, P>>,
    pub(super) backward_closure: Box<dyn Fn(Variable<T, S, C, L, P, Lgrad>) -> ()>,
}

pub struct Variable<T, S, C, L, P, Lgrad>(pub(super) Rc<RefCell<BackpropNode<T, S, C, L, P, Lgrad>>>);

impl<T, S, C, L, P, Lgrad> Deref for Variable<T, S, C, L, P, Lgrad> {
    type Target = Rc<RefCell<BackpropNode<T, S, C, L, P, Lgrad>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, S, C, L, P, Lgrad> Variable<T, S, C, L, P, Lgrad>
where
    Tensor<T, S, Contiguous, Lgrad, P>: Clone,
{
    pub fn grad(&self) -> Option<Tensor<T, S, Contiguous, Lgrad, P>> {
        self.borrow().grad.clone()
    }
}

impl<T, S, C, L, P> Variable<T, S, C, L, P, P::Layout>
where
    S: StaticShape,
    L: for<'a> Layout<'a, T>,
    P: StaticAllocationPolicy<T, S>,
{
    pub fn backward<Cback, Pback, Lback>(&self, grad: &Tensor<T, S, Cback, Lback, Pback>)
    where
        T: Send + Sync + Copy + AddAssign,
        Lback: for<'a> Layout<'a, T>,
    {
        {
            let mut node = self.borrow_mut();
            if let Some(current_grad) = &mut node.grad {
                current_grad.add_(grad);
            }
        }

        let node = self.borrow();
        (node.backward_closure)(Self::clone(&self));
    }

    pub fn new(tensor: Tensor<T, S, C, L, P>, require_grad: bool) -> Self {
        Variable(Rc::new(RefCell::new(BackpropNode {
            value: tensor,
            grad: if require_grad {
                Some(Tensor::default())
            } else {
                None
            },
            backward_closure: Box::new(|_var| ()),
        })))
    }
}

impl<T, S, C, L, P> Variable<T, S, C, L, P, P::Layout>
where
    L: for<'a> Layout<'a, T>,
    P: DynamicAllocationPolicy<T>,
{
    pub fn backward_dynamic<Cback, Lback, Pback>(&self, grad: &Tensor<T, S, Cback, Lback, Pback>)
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
                current_grad.add_dynamic_(grad);
            }
        }

        let node = self.borrow();
        (node.backward_closure)(Self::clone(&self));
    }
}

// impl<T, S> From<StaticTensor<T, S>> for Variable<T, S, Contiguous, StaticHeapLayout<T, S>, DefaultPolicy, Lgrad> {
//     fn from(tensor_like: Box<dyn TensorLike<T, S>>) -> Self {
//         Variable(Rc::new(RefCell::new(BackpropNode {
//             value: tensor_like,
//             grad: None,
//             require_grad: false,
//             backward_closure: Box::new(|| ()),
//         })))
//     }
// }

impl<T, S, C, L, P, Lgrad> Clone for Variable<T, S, C, L, P, Lgrad> {
    fn clone(&self) -> Self {
        Variable(Rc::clone(&self.0))
    }
}
