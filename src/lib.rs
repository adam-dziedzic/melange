#[cfg(test)]
mod tests {
    use super::prelude::*;
    use typenum::{U1, U2, U3, U4, U5, U6};
    use typenum::marker_traits::{Bit, Unsigned};

    #[test]
    fn shape() {
        assert_eq!(<Shape2D<U3, U2> as StaticShape>::to_vec(), vec![3, 2]);
        assert_eq!(<Shape2D<U3, U2> as StaticShape>::strides(), vec![2, 1]);
        assert_eq!(<Shape2D<U3, U2> as StaticShape>::NUM_ELEMENTS, 6);
    }

    #[test]
    fn shape_constraints() {
        assert_eq!(<Shape3D<U2, U1, U2> as Broadcast<Shape4D<U5, U1, U3, U2>>>::Output::BOOL, true);
        assert_eq!(<Shape3D<U2, U4, U2> as Broadcast<Shape4D<U5, U1, U3, U2>>>::Output::BOOL, false);
        assert_eq!(<Shape1D<U6> as SameNumElements<i32, Shape2D<U3, U2>>>::Output::BOOL, true);
        assert_eq!(<Shape1D<U6> as SameNumElements<i32, Shape2D<U3, U3>>>::Output::BOOL, false);
        assert_eq!(<Shape2D<Dyn, U2> as Same<Shape2D<U3, U2>>>::Output::BOOL, true);
        assert_eq!(<Shape2D<Dyn, U2> as Same<Shape2D<U3, U3>>>::Output::BOOL, false);
        assert_eq!(<Shape4D<U5, U1, U3, U2> as NumElements<i32>>::Output::U8, 30_u8);
        assert_eq!(<Shape2D<Dyn, U2> as ReprShape<i32, Shape2D<U3, Dyn>>>::Output::to_vec(), vec![3, 2]);
        assert!(<Shape2D<Dyn, Dyn> as Shape>::runtime_compat(&[3, 3]));
        assert!(<Shape2D<U3, Dyn> as Shape>::runtime_compat(&[3, 3]));
    }

    #[test]
    fn broadcast_same_order() {
        let a: SliceTensor<i32, Shape2D<U1, U2>> = Tensor::from_slice(&[1, 2]);
        let b: SliceTensor<_, Shape2D<U2, U2>> = a.broadcast();

        let c: SliceTensor<i32, Shape2D<U2, U2>> = Tensor::from_slice(&[1, 2, 1, 2]);
        assert_eq!(b, c);
        assert_eq!(b.shape(), vec![2, 2]);
        assert_eq!(b.strides(), vec![0, 1]);
        assert_eq!(b.opt_chunk_size(), 2);
    }

    #[test]
    fn broadcast_different_order() {
        let a: SliceTensor<i32, Shape1D<U2>> = Tensor::from_slice(&[1, 2]);
        let b: SliceTensor<_, Shape2D<U2, U2>> = a.broadcast();

        let c: SliceTensor<i32, Shape2D<U2, U2>> = Tensor::from_slice(&[1, 2, 1, 2]);
        assert_eq!(b, c);
        assert_eq!(b.shape(), vec![2, 2]);
        assert_eq!(b.strides(), vec![0, 1]);
        assert_eq!(b.opt_chunk_size(), 2);
    }

    #[test]
    fn reshape() {
        let mut a: StackTensor<i32, Shape1D<U4>> = Tensor::default();
        for (x, y) in a.iter_mut().zip(&[1, 2, 1, 2]) {
            *x = *y;
        }
        let b: SliceTensor<_, Shape2D<U2, U2>> = a.reshape();

        let c: SliceTensor<i32, Shape2D<U2, U2>> = Tensor::from_slice(&[1, 2, 1, 2]);
        assert_eq!(b, c);
        assert_eq!(b.shape(), vec![2, 2]);
        assert_eq!(b.strides(), vec![2, 1]);
        assert_eq!(b.opt_chunk_size(), 4);
    }

    #[test]
    fn add() {
        let a: SliceTensor<i32, Shape2D<U3, U3>> = Tensor::from_slice(&[1, 0, 0, 0, 1, 0, 0, 0, 1]);
        let b: SliceTensor<i32, Shape2D<U3, U3>> = Tensor::from_slice(&[1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let c = a.add(&b);

        let d: SliceTensor<i32, Shape2D<U3, U3>> = Tensor::from_slice(&[2, 1, 1, 1, 2, 1, 1, 1, 2]);
        assert_eq!(c.as_view(), d);
    }

    #[test]
    fn add_static_ok() {
        let a: SliceTensor<i32, Shape2D<Dyn, Dyn>> = Tensor::from_slice_dyn(&[1, 0, 0, 0, 1, 0, 0, 0, 1], vec![3, 3]);
        let b: SliceTensor<i32, Shape2D<U3, U3>> = Tensor::from_slice(&[1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let c = a.add_static(&b);

        let d: SliceTensor<i32, Shape2D<U3, U3>> = Tensor::from_slice(&[2, 1, 1, 1, 2, 1, 1, 1, 2]);
        assert_eq!(c.as_view(), d);

        // let t: SliceTensor<i32, Shape2D<U3, Dyn>> = Tensor::from_slice_dyn(&[1, 1, 1, 1, 1, 1, 1, 1, 1], vec![3, 3]);
        // let u: StackTensor<_, _> = a.add_dyn(&t);
    }

    #[test]
    #[should_panic(expected = "Tensors must have same shape")]
    fn add_static_panic() {
        let a: SliceTensor<i32, Shape2D<Dyn, Dyn>> = Tensor::from_slice_dyn(&[1, 0, 0, 0, 1, 0, 0, 0, 1], vec![3, 3]);
        let b: SliceTensor<i32, Shape2D<U2, U2>> = Tensor::from_slice(&[1, 1, 1, 1]);
        let _c = a.add_static(&b);
    }

    #[test]
    fn add_dyn_ok() {
        let a: SliceTensor<i32, Shape2D<Dyn, Dyn>> = Tensor::from_slice_dyn(&[1, 0, 0, 0, 1, 0, 0, 0, 1], vec![3, 3]);
        let b: SliceTensor<i32, Shape2D<U3, Dyn>> = Tensor::from_slice_dyn(&[1, 1, 1, 1, 1, 1, 1, 1, 1], vec![3, 3]);
        let c = a.add_dyn(&b);
        let c: SliceTensor<_, Shape2D<U3, U3>> = c.as_static();

        let d: SliceTensor<i32, Shape2D<U3, U3>> = Tensor::from_slice(&[2, 1, 1, 1, 2, 1, 1, 1, 2]);
        assert_eq!(c, d);
    }

    #[test]
    fn add_broadcast() {
        let a: SliceTensor<i32, Shape2D<U3, U3>> = Tensor::from_slice(&[1, 0, 0, 0, 1, 0, 0, 0, 1]);
        let b: SliceTensor<i32, Shape1D<U3>> = Tensor::from_slice(&[1, 1, 1]);
        let c: StackTensor<_, _> = a.add(&b.broadcast());

        let d: SliceTensor<i32, Shape2D<U3, U3>> = Tensor::from_slice(&[2, 1, 1, 1, 2, 1, 1, 1, 2]);
        assert_eq!(c.as_view(), d);
    }
}

pub mod tensor;
pub mod prelude;