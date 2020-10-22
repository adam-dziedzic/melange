//! `reductions` contains reduction operations at the variable level
//! that rely on the implementation of the `tensor` module.

use super::variable::{BackpropNode, Variable};
use crate::tensor::allocation_policy::StaticAllocationPolicy;
use crate::tensor::prelude::*;
use crate::tensor::transpose_policy::Contiguous;
use melange_macros::{define_closure, expand_operations};
use std::cell::RefCell;
use std::ops::*;
use std::rc::Rc;

#[expand_operations(
    sum<T: Send + Sync + Copy + AddAssign + 'static + Add<Output = T> + Default>,
    // prod<T: Send + Sync + Copy + MulAssign + 'static>,
)]
#[define_closure(
    sum: move |grad| {
        self.backward(grad.broadcast().as_contiguous());
    }
)]
impl<T, S, C, L, P, Pback> Variable<T, S, C, L, P, P::Layout, Contiguous, Pback::Layout, Pback>
where
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
{
    fn operation<Ax, Cback, Lback>(
        self,
    ) -> Variable<
        T,
        <S as Reduction<Ax>>::Output,
        Contiguous,
        <P as StaticAllocationPolicy<T, <S as Reduction<Ax>>::Output>>::Layout,
        P,
        <P as StaticAllocationPolicy<T, S>>::Layout,
        Cback,
        Lback,
        Pback,
    >
    where
        S: Reduction<Ax> + ReductionOptChunckSize<T, Ax> + At<Ax>,
        <S as Reduction<Ax>>::Output: Broadcast<S>,
        <<S as Reduction<Ax>>::Output as Broadcast<S>>::Output: TRUE,
        P: StaticAllocationPolicy<T, <S as Reduction<Ax>>::Output>,
        <P as StaticAllocationPolicy<T, <S as Reduction<Ax>>::Output>>::Layout:
            for<'a> Layout<'a, T> + 'static,
        <S as Reduction<Ax>>::Output: StaticShape,
        Cback: 'static,
        Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
    {
        let (value, grad) = {
            let self_ref = self.borrow();
            (
                self_ref.value.placeholder::<Ax>(),
                if let Some(_) = self_ref.grad {
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
            backward_closure: Box::new(|| ()),
        })))
    }
}
