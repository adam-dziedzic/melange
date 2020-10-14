use super::ring::Ring;
use super::variable::{BackpropNode, Variable};
use crate::tensor::allocation_policy::StaticAllocationPolicy;
use crate::tensor::prelude::*;
use crate::tensor::transpose_policy::Contiguous;
use road_ai_macros::{define_closure, expand_operations};
use std::cell::RefCell;
use std::ops::*;
use std::rc::Rc;

#[expand_operations(
    add<T: Send + Sync + Copy + AddAssign + Add<Output = T> + 'static> in Add,
    sub<T: Send + Sync + Copy + AddAssign + Sub<Output = T> + Ring<T> + Neg<Output = T> + Mul<Output = T> + 'static> in Sub,
    mul<T: Send + Sync + Copy + AddAssign + MulAssign + Mul<Output = T> + 'static> in Mul,
    div<T: Send + Sync + Copy + AddAssign + DivAssign + Div<Output = T> + Mul<Output = T> + 'static> in Div,
)]
#[define_closure(
    add: move |grad| {
        let mut other_grad = grad.clone();
        self.backward(grad);
        other.backward(&mut other_grad);
    }
)]
#[define_closure(
    sub: move |grad| {
        let mut other_grad = grad.scal_mul(-T::ONE);
        self.backward(grad);
        other.backward(&mut other_grad);
    }
)]
#[define_closure(
    mul: move |grad| {
        let mut other_grad = {
            let self_ref = self.borrow();
            grad.mul(&self_ref.value)
        };
        
        {
            let other_ref = other.borrow();
            grad.mul_(&other_ref.value);
        }

        self.backward(grad);
        other.backward(&mut other_grad);
    }
)]
#[define_closure(
    div: move |grad| {
        let mut other_grad = {
            let self_ref = self.borrow();
            grad.mul(&self_ref.value)
        };

        {
            let other_ref = other.borrow();
            grad.div_(&other_ref.value)
        }

        self.backward(grad);
        other.backward(&mut other_grad);
    }
)]
impl<T, S, C, L, P, Lback, Pback, Crhs, Lrhs, Prhs>
    ImplTrait<Variable<T, S, Crhs, Lrhs, Prhs, Prhs::Layout, Pback::Layout, Pback>> for Variable<T, S, C, L, P, P::Layout, Lback, Pback>
where
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Crhs: 'static,
    Prhs: StaticAllocationPolicy<T, S> + 'static,
    Prhs::Layout: for<'a> Layout<'a, T> + 'static,
    Lrhs: for<'a> Layout<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + 'static,
    Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
{
    type Output = Variable<T, S, Contiguous, P::Layout, P, P::Layout, Lback, Pback>;
    fn operation(self, other: Variable<T, S, Crhs, Lrhs, Prhs, Prhs::Layout, Pback::Layout, Pback>) -> Self::Output {
        let (value, grad) = {
            let self_ref = &self.borrow();
            let other_ref = &other.borrow();
            
            (
                self_ref.value.placeholder(&other_ref.value),
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
            backward_closure: Box::new(|| ()),
        })))
    }
}

#[expand_operations(
    scal_add<T: Send + Sync + Copy + AddAssign + Add<Output = T> + 'static> as add in Add,
    scal_sub<T: Send + Sync + Copy + AddAssign + Sub<Output = T> + 'static> as sub in Sub,
    scal_mul<T: Send + Sync + Copy + AddAssign + Mul<Output = T> + 'static> as mul in Mul,
    scal_div<T: Send + Sync + Copy + AddAssign + Div<Output = T> + Mul<Output = T> + Ring<T> + 'static> as div in Div,
)]
#[define_closure(
    add: move |grad| {
        self.backward(grad);
    }
)]
#[define_closure(
    sub: move |grad| {
        self.backward(grad);
    }
)]
#[define_closure(
    mul: move |grad| {
        grad.scal_mul_(param);
        self.backward(grad);
    }
)]
#[define_closure(
    div: move |grad| {
        grad.scal_mul_(T::ONE / param);
        self.backward(grad);
    }
)]
impl<T, S, C, L, P, Lback, Pback>
    ImplTrait<T> for Variable<T, S, C, L, P, P::Layout, Lback, Pback>
where
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + 'static,
    Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
{
    type Output = Variable<T, S, Contiguous, P::Layout, P, P::Layout, Lback, Pback>;
    fn operation(self, param: T) -> Self::Output {
        let (value, grad) = {
            let self_ref = &self.borrow();
            (
                self_ref.value.placeholder(param),
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

// #[define_closure(
//     powf: move |grad| {
//         let self_grad = {
//             let self_ref = self.borrow();
//             self_ref.value.powf(param - 1.0).scal_mul(param)
//         }; 
        
//         self.backward(&grad.mul(&self_grad));
//     }
// )]
// #[define_closure(
//     powi: move |grad| {
//         let self_grad = {
//             let self_ref = self.borrow();
//             self_ref.value.powi(param - 1).scal_mul(param.into())
//         }; 
        
//         self.backward(&grad.mul(&self_grad));
//     }
// )]

// #[define_closure(
//     exp: move |grad| {
//         let self_grad = {
//             let self_ref = self.borrow();
//             self_ref.value
//         }; 
        
//         self.backward(&grad.mul(&self_grad));
//     }
// )]
// #[define_closure(
//     exp2: move |grad| {
//         let self_grad = {
//             let self_ref = self.borrow();
//             self_ref.value.scal_mul(std::f64::consts::LN_2.into())
//         }; 
        
//         self.backward(&grad.mul(&self_grad));
//     }
// )]
// #[define_closure(
//     exp_m1: move |grad| {
//         let self_grad = {
//             let self_ref = self.borrow();
//             self_ref.value
//         };
        

//         self.backward(&grad.mul(&self_grad));
//     }
// )]
// #[define_closure(
//     ln: move |grad| {
//         let self_grad = {
//             let self_ref = self.borrow();
//             1.0 / 
//         };
        
//         self.backward(&grad.mul(&self_grad));
//     }
// )]
