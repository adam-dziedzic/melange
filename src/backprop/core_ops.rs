use super::variable::{BackpropNode, Variable};
use crate::tensor::allocation_policy::StaticAllocationPolicy;
use crate::tensor::prelude::*;
use crate::tensor::transpose_policy::Contiguous;
use std::cell::RefCell;
use std::ops::*;
use std::rc::Rc;

impl<T, S, C, L, P, Crhs, Lrhs, Prhs, Lgradrhs>
    Add<Variable<T, S, Crhs, Lrhs, Prhs, Lgradrhs>> for Variable<T, S, C, L, P, P::Layout>
where
    T: Send + Sync + Copy + Add<Output = T> + AddAssign + From<f32> + 'static,
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Lrhs: for<'a> Layout<'a, T>,
{
    type Output = Variable<T, S, Contiguous, P::Layout, P, P::Layout>;
    fn add(self, other: Variable<T, S, Crhs, Lrhs, Prhs, Lgradrhs>) -> Self::Output {
        let (value, grad) = {
            let self_ref = &self.borrow();
            let other_ref = &other.borrow();
            
            (
                self_ref.value.add(&other_ref.value),
                if let Some(_) = self_ref.grad {
                    Some(Tensor::default())
                } else if let Some(_) = other_ref.grad {
                    Some(Tensor::default())
                } else {
                    None
                },
            )
        };

        Variable(Rc::new(RefCell::new(BackpropNode {
            value,
            grad,
            backward_closure: Box::new(move |var| {
                if let Some(grad) = &var.borrow().grad {
                    self.backward(grad);
                }
            }),
        })))
    }
}
