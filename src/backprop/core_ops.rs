use crate::ring::Ring;
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
    sub<T: Send + Sync + Copy + AddAssign + Sub<Output = T> + Ring + Neg<Output = T> + Mul<Output = T> + 'static> in Sub,
    mul<T: Send + Sync + Copy + AddAssign + MulAssign + Mul<Output = T> + 'static> in Mul,
    div<T: Send + Sync + Copy + AddAssign + DivAssign + Div<Output = T> + Mul<Output = T> + 'static> in Div,
)]
#[define_closure(
    add: move |grad| {
        let other_grad = grad.as_contiguous();
        self.backward(grad);
        other.backward(other_grad);
    }
)]
#[define_closure(
    sub: move |grad| {
        let other_grad = grad.scal_mul(-T::ONE);
        self.backward(grad);
        other.backward(other_grad);
    }
)]
#[define_closure(
    mul: move |mut grad| {
        let other_grad = {
            let self_ref = self.borrow();
            grad.mul(&self_ref.value)
        };
        
        {
            let other_ref = other.borrow();
            grad.mul_(&other_ref.value);
        }

        self.backward(grad);
        other.backward(other_grad);
    }
)]
#[define_closure(
    div: move |mut grad| {
        let other_grad = {
            let self_ref = self.borrow();
            grad.mul(&self_ref.value)
        };

        {
            let other_ref = other.borrow();
            grad.div_(&other_ref.value)
        }

        self.backward(grad);
        other.backward(other_grad);
    }
)]
impl<T, S, C, L, P, Cback, Lback, Pback, Crhs, Lrhs, Prhs>
    ImplTrait<Variable<T, S, Crhs, Lrhs, Prhs, Prhs::Layout, Contiguous, Pback::Layout, Pback>> for Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
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
    Cback: 'static,
    Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + 'static,
{
    type Output = Variable<T, S, Contiguous, P::Layout, P, P::Layout, Cback, Lback, Pback>;
    fn operation(self, other: Variable<T, S, Crhs, Lrhs, Prhs, Prhs::Layout, Contiguous, Pback::Layout, Pback>) -> Self::Output {
        let (value, grad) = {
            let self_ref = self.borrow();
            let other_ref = other.borrow();
            
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
    scal_div<T: Send + Sync + Copy + AddAssign + Div<Output = T> + Mul<Output = T> + Ring + 'static> as div in Div,
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
    mul: move |mut grad| {
        grad.scal_mul_(param);
        self.backward(grad);
    }
)]
#[define_closure(
    div: move |mut grad| {
        grad.scal_mul_(T::ONE / param);
        self.backward(grad);
    }
)]
impl<T, S, C, L, P, Cback, Lback, Pback>
    ImplTrait<T> for Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
where
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Cback: 'static,
    Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + 'static,
{
    type Output = Variable<T, S, Contiguous, P::Layout, P, P::Layout, Cback, Lback, Pback>;
    fn operation(self, param: T) -> Self::Output {
        let (value, grad) = {
            let self_ref = self.borrow();
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

#[expand_operations(
    powf<T=f64>(f64),
    powi<T=f64>(i32),
    powf<T=f32>(f32),
    // powi<T=f32>(i32), -> no easy conversion from i32 into f32
)]
#[define_closure(
    powf: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.powf(param - 1.0)
        };
        self_grad.scal_mul_(param);
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    powi: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.powi(param - 1)
        };
        self_grad.scal_mul_(param.into());
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
impl<T, S, C, L, P, Cback, Lback, Pback> Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
where
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Cback: 'static,
    Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + 'static,
{
    fn operation(self, param: type0) -> Variable<T, S, Contiguous, P::Layout, P, P::Layout, Cback, Lback, Pback> {
        let (value, grad) = {
            let self_ref = self.borrow();
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

#[expand_operations(
    exp<T=f64>,
    exp2<T=f64>,
    exp_m1<T=f64>,
    ln<T=f64>,
    ln_1p<T=f64>,
    log2<T=f64>,
    log10<T=f64>,
    sin<T=f64>,
    cos<T=f64>,
    tan<T=f64>,
    sinh<T=f64>,
    cosh<T=f64>,
    tanh<T=f64>,
    asin<T=f64>,
    acos<T=f64>,
    atan<T=f64>,
    asinh<T=f64>,
    acosh<T=f64>,
    atanh<T=f64>,
    sqrt<T=f64>,
    cbrt<T=f64>,
    abs<T=f64>,
    exp<T=f32>,
    exp2<T=f32>,
    exp_m1<T=f32>,
    ln<T=f32>,
    ln_1p<T=f32>,
    log2<T=f32>,
    log10<T=f32>,
    sin<T=f32>,
    cos<T=f32>,
    tan<T=f32>,
    sinh<T=f32>,
    cosh<T=f32>,
    tanh<T=f32>,
    asin<T=f32>,
    acos<T=f32>,
    atan<T=f32>,
    asinh<T=f32>,
    acosh<T=f32>,
    atanh<T=f32>,
    sqrt<T=f32>,
    cbrt<T=f32>,
    abs<T=f32>,
)]
#[define_closure(
    exp: move |mut grad| {
        {
            let self_ref = self.borrow();
            grad.mul_(&self_ref.value);
        } 
        
        self.backward(grad);
    }
)]
#[define_closure(
    exp2: move |mut grad| {
        let self_grad = {
            let self_ref = self.borrow();
            self_ref.value.scal_mul(std::T::consts::LN_2)
        }; 
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    exp_m1: move |mut grad| {
        {
            let self_ref = self.borrow();
            grad.mul_(&self_ref.value);
        }
        
        self.backward(grad);
    }
)]
#[define_closure(
    ln: move |mut grad| {
        let self_grad = {
            let self_ref = self.borrow();
            self_ref.value.inv()
        };
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    ln_1p: move |mut grad| {
        let self_grad = {
            let self_ref = self.borrow();
            self_ref.value.inv()
        };
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    log2: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.inv()
        };
        self_grad.scal_div_(std::T::consts::LN_2);
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    log10: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.inv()
        };
        self_grad.scal_div_(std::T::consts::LN_10);
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    sin: move |mut grad| {
        let self_grad = {
            let self_ref = self.borrow();
            self_ref.value.cos()
        };
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    cos: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.sin()
        };
        self_grad.scal_mul_(-1.0);
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    tan: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.tan()
        };
        self_grad.powi_(2);
        self_grad.scal_add_(1.0);
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    sinh: move |mut grad| {
        let self_grad = {
            let self_ref = self.borrow();
            self_ref.value.cosh()
        };
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    cosh: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.sinh()
        };
        self_grad.scal_mul_(-1.0);
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    tanh: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.tanh()
        };
        self_grad.powi_(2);
        self_grad.scal_mul_(-1.0);
        self_grad.scal_add_(1.0);
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    asin: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.powi(2)
        };
        self_grad.scal_mul_(-1.0);
        self_grad.scal_add_(1.0);
        self_grad.sqrt_();
        self_grad.inv_();
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    acos: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.powi(2)
        };
        self_grad.scal_mul_(-1.0);
        self_grad.scal_add_(1.0);
        self_grad.sqrt_();
        self_grad.inv_();
        self_grad.scal_mul_(-1.0);
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    atan: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.powi(2)
        };
        self_grad.scal_add_(1.0);
        self_grad.inv_();
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    asinh: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.powi(2)
        };
        self_grad.scal_add_(1.0);
        self_grad.sqrt_();
        self_grad.inv_();
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    acosh: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.powi(2)
        };
        self_grad.scal_add_(-1.0);
        self_grad.sqrt_();
        self_grad.inv_();
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    atanh: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.powi(2)
        };
        self_grad.scal_mul_(-1.0);
        self_grad.scal_add_(1.0);
        self_grad.inv_();
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    sqrt: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.sqrt()
        };
        self_grad.scal_mul_(2.0);
        self_grad.inv_();
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    cbrt: move |mut grad| {
        let mut self_grad = {
            let self_ref = self.borrow();
            self_ref.value.cbrt()
        };
        self_grad.scal_mul_(3.0);
        self_grad.inv_();
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
#[define_closure(
    abs: move |mut grad| {
        let self_grad = {
            let self_ref = self.borrow();
            self_ref.value.signum()
        };
        
        grad.mul_(&self_grad);
        self.backward(grad);
    }
)]
impl<T, S, C, L, P, Cback, Lback, Pback> Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
where
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Cback: 'static,
    Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + 'static,
{
    fn operation(self) -> Variable<T, S, Contiguous, P::Layout, P, P::Layout, Cback, Lback, Pback> {
        let (value, grad) = {
            let self_ref = self.borrow();
            (
                self_ref.value.placeholder(),
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

#[expand_operations(
    scal_mul_add<T=f64>(f64, f64),
    scal_mul_add<T=f32>(f32, f32),
)]
#[define_closure(
    scal_mul_add: move |mut grad| {
        grad.scal_mul_(param0);
        self.backward(grad);
    }
)]
impl<T, S, C, L, P, Cback, Lback, Pback> Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
where
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Cback: 'static,
    Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + 'static,
{
    fn operation(self, param0: type0, param1: type1) -> Variable<T, S, Contiguous, P::Layout, P, P::Layout, Cback, Lback, Pback> {
        let (value, grad) = {
            let self_ref = self.borrow();
            (
                self_ref.value.placeholder(param0, param1),
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

#[expand_operations(
    mul_add<T=f64>,
    mul_add<T=f32>,
)]
#[define_closure(
    mul_add: move |mut grad| {
        let other0_grad = {
            let self_ref = self.borrow();
            grad.mul(&self_ref.value)
        };

        let other1_grad = grad.as_contiguous();
        
        {
            let other0_ref = self.borrow();
            grad.mul_(&other0_ref.value);
        }
        
        self.backward(grad);
        other0.backward(other0_grad);
        other1.backward(other1_grad);
    }
)]
impl<T, S, C, L, P, Cback, Lback, Pback> Variable<T, S, C, L, P, P::Layout, Cback, Lback, Pback>
where
    S: StaticShape + 'static,
    C: 'static,
    L: for<'a> Layout<'a, T> + 'static,
    P: StaticAllocationPolicy<T, S> + 'static,
    P::Layout: for<'a> Layout<'a, T> + 'static,
    Cback: 'static,
    Lback: for<'a> Layout<'a, T> + for<'a> LayoutMut<'a, T> + 'static,
    Pback: StaticAllocationPolicy<T, S> + 'static,
    Pback::Layout: for<'a> Layout<'a, T> + 'static,
{
    fn operation<Crhs0, Lrhs0, Prhs0, Crhs1, Lrhs1, Prhs1>(self, other0: Variable<T, S, Crhs0, Lrhs0, Prhs0, Prhs0::Layout, Contiguous, Pback::Layout, Pback>, other1: Variable<T, S, Crhs1, Lrhs1, Prhs1, Prhs1::Layout, Contiguous, Pback::Layout, Pback>) -> Variable<T, S, Contiguous, P::Layout, P, P::Layout, Cback, Lback, Pback>
    where
        Crhs0: 'static,
        Lrhs0: for<'a> Layout<'a, T> + 'static,
        Prhs0: StaticAllocationPolicy<T, S> + 'static,
        Prhs0::Layout: for<'a> Layout<'a, T> + 'static,
        Crhs1: 'static,
        Lrhs1: for<'a> Layout<'a, T> + 'static,
        Prhs1: StaticAllocationPolicy<T, S> + 'static,
        Prhs1::Layout: for<'a> Layout<'a, T> + 'static,
    {
        let (value, grad) = {
            let self_ref = self.borrow();
            let other0_ref = other0.borrow();
            let other1_ref = other1.borrow();
            (
                self_ref.value.placeholder(&other0_ref.value, &other1_ref.value),
                if let Some(_) = self_ref.grad {
                    Some(Tensor::default())
                } else if let Some(_) = other0_ref.grad {
                    Some(Tensor::default())
                } else if let Some(_) = other1_ref.grad {
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
