#[cfg(test)]
mod tests {
    use super::prelude::*;
    use typenum::marker_traits::{Bit, Unsigned};
    use typenum::{U0, U1, U2, U3, U4, U5, U6};

    #[test]
    fn shape() {
        assert_eq!(<Shape2D<U3, U2> as StaticShape>::to_vec(), vec![3, 2]);
        assert_eq!(<Shape2D<U3, U2> as StaticShape>::strides(), vec![2, 1]);
        assert_eq!(<Shape2D<U3, U2> as StaticShape>::NUM_ELEMENTS, 6);
    }

    #[test]
    fn shape_constraints() {
        assert_eq!(
            <Shape3D<U2, U1, U2> as Broadcast<Shape4D<U5, U1, U3, U2>>>::Output::BOOL,
            true
        );
        assert_eq!(
            <Shape3D<U2, U4, U2> as Broadcast<Shape4D<U5, U1, U3, U2>>>::Output::BOOL,
            false
        );
        assert_eq!(
            <Shape1D<U6> as SameNumElements<i32, Shape2D<U3, U2>>>::Output::BOOL,
            true
        );
        assert_eq!(
            <Shape1D<U6> as SameNumElements<i32, Shape2D<U3, U3>>>::Output::BOOL,
            false
        );
        assert_eq!(
            <Shape2D<Dyn, U2> as Same<Shape2D<U3, U2>>>::Output::BOOL,
            true
        );
        assert_eq!(
            <Shape2D<Dyn, U2> as Same<Shape2D<U3, U3>>>::Output::BOOL,
            false
        );
        assert_eq!(
            <Shape4D<U5, U1, U3, U2> as NumElements<i32>>::Output::USIZE,
            30
        );
        assert_eq!(
            <Shape2D<Dyn, U2> as ReprShape<i32, Shape2D<U3, Dyn>>>::Output::to_vec(),
            vec![3, 2]
        );
        assert!(<<Shape2D<Dyn, Dyn> as ReprShapeDyn<i32, Shape2D<U2, Dyn>>>::Output as Shape>::runtime_compat(&[2, 3]));
        assert!(<<Shape2D<U2, Dyn> as ReprShapeDyn<i32, Shape2D<Dyn, Dyn>>>::Output as Shape>::runtime_compat(&[2, 3]));
        assert!(<<Shape2D<Dyn, Dyn> as ReprShapeDyn<
            i32,
            Shape2D<Dyn, Dyn>,
        >>::Output as Shape>::runtime_compat(&[3, 3]));
        assert!(<Shape2D<Dyn, Dyn> as Shape>::runtime_compat(&[3, 3]));
        assert!(<Shape2D<U3, Dyn> as Shape>::runtime_compat(&[3, 3]));
        assert_eq!(
            <<Shape2D<U2, U3> as Reduction<U1>>::Output as StaticShape>::to_vec(),
            vec![2, 1]
        );
        assert_eq!(
            <Shape2D<U2, U3> as ReductionOptChunckSize<i32, U0>>::Output::USIZE,
            3
        );
        assert_eq!(
            <Shape2D<U3, U3> as ReductionOptChunckSize<i32, U1>>::Output::USIZE,
            1
        );
        assert_eq!(<Shape4D<U5, U1, U3, U2> as At<U2>>::Output::USIZE, 3);
        assert_eq!(
            <<Shape4D<U4, U3, U6, U6> as Transpose>::Output as StaticShape>::to_vec(),
            vec![6, 6, 3, 4]
        );
    }

    #[test]
    fn broadcast_same_order() {
        let a: StaticTensor<i32, Shape2D<U1, U2>> = Tensor::try_from(vec![1, 2]).unwrap();
        let b: StridedStaticSliceTensor<_, Shape2D<U2, U2>> = a.broadcast();

        let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 1, 2]).unwrap();
        assert_eq!(b, c.stride::<Shape2D<U1, U1>>());
        assert_eq!(b.shape(), vec![2, 2]);
        assert_eq!(b.strides(), vec![0, 1]);
        assert_eq!(b.opt_chunk_size(), 2);
    }

    #[test]
    fn broadcast_different_order() {
        let a: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![1, 2]).unwrap();
        let b: StridedStaticSliceTensor<_, Shape2D<U2, U2>> = a.broadcast();

        let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 1, 2]).unwrap();
        assert_eq!(b, c.stride::<Shape2D<U1, U1>>());
        assert_eq!(b.shape(), vec![2, 2]);
        assert_eq!(b.strides(), vec![0, 1]);
        assert_eq!(b.opt_chunk_size(), 2);
    }

    #[test]
    fn reshape() {
        let a: StaticTensor<i32, Shape1D<U4>> = Tensor::try_from(vec![1, 2, 1, 2]).unwrap();
        let b: StaticSliceTensor<_, Shape2D<U2, U2>> = a.reshape();

        let c: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 1, 2]).unwrap();
        assert_eq!(b, c.as_view());
        assert_eq!(b.shape(), vec![2, 2]);
        assert_eq!(b.strides(), vec![2, 1]);
        assert_eq!(b.opt_chunk_size(), 4);
    }

    #[test]
    fn add() {
        let a: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        let b: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let c = a.add(&b);

        let d: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![2, 1, 1, 1, 2, 1, 1, 1, 2]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn mul() {
        let a: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1, 0, 0, 0, 3, 0, 0, 0, 1]).unwrap();
        let b: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![3, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let c = a.mul(&b);

        let d: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![3, 0, 0, 0, 3, 0, 0, 0, 1]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn add_coerce_ok() {
        let a: DynamicTensor<i32, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        let b: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let c = a.add(&b);

        let d: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![2, 1, 1, 1, 2, 1, 1, 1, 2]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    #[should_panic(expected = "Tensors must have same shape")]
    fn add_static_panic() {
        let a: DynamicTensor<i32, Shape2D<Dyn, U2>> =
            Tensor::try_from(vec![1, 0, 0, 0, 1, 0, 0, 0]).unwrap();
        let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 1, 1, 1]).unwrap();
        let _c = a.add(&b);
    }

    #[test]
    fn add_dyn_ok() {
        let a: DynamicTensor<i32, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        let b: DynamicTensor<i32, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let c = a.add_dynamic(&b);

        let d: DynamicTensor<i32, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![2, 1, 1, 1, 2, 1, 1, 1, 2]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn add_broadcast() {
        let a: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        let b: StaticTensor<i32, Shape1D<U3>> = Tensor::try_from(vec![1, 1, 1]).unwrap();
        let c: StaticTensor<_, _> = a.add(&b.broadcast());

        let d: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![2, 1, 1, 1, 2, 1, 1, 1, 2]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn scal_add() {
        let a: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        let c = a.scal_add(1);

        let d: StaticTensor<i32, Shape2D<U3, U3>> =
            Tensor::try_from(vec![2, 1, 1, 1, 2, 1, 1, 1, 2]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn scal_add_dyn() {
        let a: DynamicTensor<i32, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        let c = a.scal_add(1);

        let d: DynamicTensor<i32, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![2, 1, 1, 1, 2, 1, 1, 1, 2]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn exp() {
        let a: StaticTensor<f64, Shape2D<U3, U3>> = Tensor::try_from(vec![
            2.0_f64.ln(),
            0.0,
            0.0,
            0.0,
            2.0_f64.ln(),
            0.0,
            0.0,
            0.0,
            2.0_f64.ln(),
        ])
        .unwrap();
        let c = a.exp();

        let d: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn exp_dyn() {
        let a: DynamicTensor<f64, Shape2D<Dyn, U3>> = Tensor::try_from(vec![
            2.0_f64.ln(),
            0.0,
            0.0,
            0.0,
            2.0_f64.ln(),
            0.0,
            0.0,
            0.0,
            2.0_f64.ln(),
        ]).unwrap();
        let c = a.exp();

        let d: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn powi() {
        let a: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]).unwrap();
        let c = a.powi(2);

        let d: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![4.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn powi_dyn() {
        let a: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]).unwrap();
        let c = a.powi(2);

        let d: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![4.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn max() {
        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 5.0, 9.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, 1.0, 0.0, 1.0]).unwrap();
        let c = a.max(&b);

        let d: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, 1.0, 5.0, 9.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn max_static() {
        let a: DynamicTensor<f64, Shape2D<Dyn, U2>> =
            Tensor::try_from(vec![1.0, 0.0, 5.0, 9.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U2, U2>> =
            Tensor::try_from(vec![3.0, 1.0, 0.0, 1.0]).unwrap();
        let c = a.max(&b);

        let d: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![3.0, 1.0, 5.0, 9.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn max_dyn() {
        let a: DynamicTensor<f64, Shape2D<Dyn, U2>> =
            Tensor::try_from(vec![1.0, 0.0, 5.0, 9.0]).unwrap();
        let b: DynamicTensor<f64, Shape2D<Dyn, U2>> =
            Tensor::try_from(vec![3.0, 1.0, 0.0, 1.0]).unwrap();
        let c = a.max_dynamic(&b);

        let d: DynamicTensor<f64, Shape2D<Dyn, U2>> = Tensor::try_from(vec![3.0, 1.0, 5.0, 9.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn mul_add() {
        let a: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let x: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let c = a.mul_add(&x, &b);

        let d: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn mul_add_static() {
        let a: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let x: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let c = a.mul_add(&x, &b);

        let d: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn mul_add_dyn() {
        let a: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let x: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]).unwrap();
        let b: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let c = a.mul_add_dynamic(&x, &b);

        let d: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn scal_mal_add() {
        let a: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let c = a.scal_mul_add(2.0, 1.0);

        let d: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn scal_mul_add_dyn() {
        let a: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let c = a.scal_mul_add(2.0, 1.0);

        let d: DynamicTensor<f64, Shape2D<Dyn, U3>> =
            Tensor::try_from(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn sum() {
        let a: StaticTensor<f64, Shape2D<U3, U3>> =
            Tensor::try_from(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let c = a.sum::<U1>();

        let d: StaticTensor<f64, Shape2D<U3, U1>> = Tensor::try_from(vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn inverse_dot() {
        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, -1.0, 0.0, 1.0]).unwrap();
        let c = a.dot(&b);

        let d: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn rotation_dot() {
        use std::f64::consts::FRAC_PI_4;
        use std::f64::EPSILON;
        let a: StaticTensor<f64, Shape2D<U3, U3>> = Tensor::try_from(vec![FRAC_PI_4.cos(), -FRAC_PI_4.sin(), 0.0, FRAC_PI_4.sin(), FRAC_PI_4.cos(), 0.0, 0.0, 0.0, 1.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U3, U1>> = Tensor::try_from(vec![1.0, 0.0, 3.0]).unwrap();
        let c = a.dot(&b);
        let t: StaticTensor<f64, Shape2D<U3, U1>> = Tensor::try_from(vec![FRAC_PI_4.cos(), FRAC_PI_4.cos(), 3.0]).unwrap();
        let d = c.sub(&t).sum::<U0>().chunks(1).nth(0).unwrap()[0];
        assert!(d < EPSILON);
    }

    #[test]
    fn transpose() {
        let a: StaticTensor<i32, Shape2D<U2, U3>> = Tensor::try_from(vec![1, 2, 3, 4, 5, 6]).unwrap();
        let b = a.transpose();

        let c: StaticTensor<i32, Shape2D<U3, U2>> = Tensor::try_from(vec![1, 4, 2, 5, 3, 6]).unwrap();
        let d = b.sub(&c).sum::<U0>().sum::<U1>().chunks(1).nth(0).unwrap()[0];
        assert_eq!(d, 0);
        assert_eq!(b.shape(), vec![3, 2]);
        assert_eq!(b.strides(), vec![1, 3]);
        assert_eq!(b.opt_chunk_size(), 1);
    }

    #[test]
    fn backprop() {
        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 1.0, 0.0, 1.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 1.0, 0.0, 1.0]).unwrap();

        let a = Variable::new(a, true);
        let b = Variable::new(b, false);

        let c: Variable<_, _, _, _, _, _, _, _, _, _, _, _> = Variable::clone(&a) + b;
        c.backward(StaticTensor::fill(1.0));
        assert_eq!(a.grad().unwrap(), StaticTensor::fill(1.0));
    }

    #[test]
    fn backprop2() {
        let a: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, 1.0, 0.0, 2.0]).unwrap();
        let c: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        let a = Variable::new(a, true);
        let b = Variable::new(b, false);
        let c = Variable::new(c, false);

        let a_times_b = Variable::clone(&a) * b;
        let result = a_times_b + c;
        result.backward(StaticTensor::fill(1.0));

        let d: StaticTensor<f64, Shape2D<U2, U2>> = Tensor::try_from(vec![2.0, 1.0, 0.0, 2.0]).unwrap();
        assert_eq!(a.grad().unwrap(), d);
    }
}

pub mod backprop;
pub mod prelude;
pub mod ring;
pub mod tensor;
