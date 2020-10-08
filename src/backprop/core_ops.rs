use super::variable::{BackpropNode, Variable};
use crate::tensor::allocation_policy::StaticAllocationPolicy;
use crate::tensor::prelude::*;
use crate::tensor::transpose_policy::Contiguous;
use std::cell::RefCell;
use std::ops::*;
use std::rc::Rc;

impl<T, S, C, L, P, Cback, Lback, Pback, Crhs, Lrhs, Prhs>
    Add<Variable<T, S, Crhs, Lrhs, Prhs, Prhs::Layout, Cback, Lback, Pback>> for Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
where
    T: Send + Sync + Copy + Add<Output = T> + AddAssign + From<f32> + 'static,
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Crhs: 'static,
    Prhs: StaticAllocationPolicy<T, S> + 'static,
    Prhs::Layout: for<'a> Layout<'a, T> + 'static,
    Lrhs: for<'a> Layout<'a, T> + 'static,
    Cback: 'static,
    Pback: 'static,
    Lback: for<'a> Layout<'a, T> + 'static,
{
    type Output = Variable<T, S, Contiguous, P::Layout, P, P::Layout, Cback, Lback, Pback>;
    fn add(self, other: Variable<T, S, Crhs, Lrhs, Prhs, Prhs::Layout, Cback, Lback, Pback>) -> Self::Output {
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
            backward_op_name: "add_back",
            backward_closure: Box::new(move |grad| {
                self.backward(grad);
                other.backward(grad);
            }),
        })))
    }
}
