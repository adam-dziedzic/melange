use proc_macro2::Ident;
use std::iter::FromIterator;
use syn::punctuated::Punctuated;
use syn::visit_mut::{self, VisitMut};
use syn::{
    Expr, ExprCall, ExprClosure, ExprMethodCall, ExprPath, GenericParam, Generics, PatType, Type,
};

pub struct FindReplaceExprMethodCall {
    pub find: Ident,
    pub replace: Ident,
}

pub struct FindReplaceType {
    pub find: Ident,
    pub replace: Type,
}

pub struct RemoveGenerics {
    pub find: Ident,
}

pub struct FindReplacePatType {
    pub find: Ident,
    pub replace: Type,
}

pub struct ReplaceExprClosure {
    pub replace: ExprClosure,
}

impl VisitMut for FindReplaceExprMethodCall {
    fn visit_expr_method_call_mut(&mut self, node: &mut ExprMethodCall) {
        if node.method == self.find {
            node.method = self.replace.clone();
        }
        visit_mut::visit_expr_method_call_mut(self, node);
    }

    fn visit_expr_call_mut(&mut self, node: &mut ExprCall) {
        if let Expr::Path(func_path) = &*node.func {
            if func_path.path.is_ident(&self.find) {
                node.func = Box::new(Expr::Path(ExprPath {
                    attrs: Vec::new(),
                    qself: None,
                    path: self.replace.clone().into(),
                }));
            }
        }
        visit_mut::visit_expr_call_mut(self, node);
    }
}

impl VisitMut for FindReplaceType {
    fn visit_type_mut(&mut self, node: &mut Type) {
        if let Type::Path(type_path) = &node {
            if type_path.path.is_ident(&self.find) {
                *node = self.replace.clone();
            }
        }

        visit_mut::visit_type_mut(self, node);
    }
}

impl VisitMut for RemoveGenerics {
    fn visit_generics_mut(&mut self, node: &mut Generics) {
        node.params = Punctuated::from_iter(
            node.params
                .iter()
                .filter(|x| {
                    if let GenericParam::Type(type_param) = &x {
                        return type_param.ident != self.find;
                    }

                    true
                })
                .cloned(),
        );

        visit_mut::visit_generics_mut(self, node);
    }
}

impl VisitMut for FindReplacePatType {
    fn visit_pat_type_mut(&mut self, node: &mut PatType) {
        if let Type::Path(type_path) = &*node.ty {
            if type_path.path.is_ident(&self.find) {
                node.ty = Box::new(self.replace.clone());
            }
        }

        visit_mut::visit_pat_type_mut(self, node);
    }
}

impl VisitMut for ReplaceExprClosure {
    fn visit_expr_closure_mut(&mut self, node: &mut ExprClosure) {
        *node = self.replace.clone();

        visit_mut::visit_expr_closure_mut(self, node);
    }
}
