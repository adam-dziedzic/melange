//! `backprop` extends the tensor module with the concept of
//! variables that are structs containing a value tensor, an option
//! for a gradient tensor and a move closure that keeps track of the
//! operands that created the value and backpropagates the gradient.
//! 
//! Most of the operations available on tensors and that are differentiable
//! are implemented on variables with the appropriate backprop closure.
//! However, unlike with tensors, variable operations move their operands
//! which is a way to ensure that variable are correctly used and allows
//! to overload common operators.
//! 
//! Under the hood, variables are a combination of `Rc` and `RefCell` that
//! allow mutable reference which fundamental to construct acyclic computation
//! graphs. Because operations on variables utilise the move mechanism, creating
//! cycles becomes harder and the user is forced to explicitly clone the `Rc`
//! to use it twice. Variables automatically dereference to their inner `Rc`
//! for ease of use.
//! 
//! Variables can retain or not their gradient during backpropagation in order
//! to save memory. Gradient retention is determined at the creation of the varaible.
//! Backpropagation closure make extensive use of in-place operations in order to
//! reduce the memory footprint. This unfortunately means that Melange is not able
//! to compute second higher order derivatives yet.
//! 
//! Note that `Variable` has generic the type parameters `Cback`, `Lback` and `Pback`
//! that correspond to the transpose policy, layout and allocation policy of the
//! backpropagated gradient. Those parameters cannot be inferred by the compiler
//! unless the computation graph is complete and a backpropagation is performed.

pub mod core_ops;
pub mod prelude;
pub mod reduction;
pub mod variable;
