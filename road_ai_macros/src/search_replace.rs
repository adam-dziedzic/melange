use std::iter::FromIterator;
use proc_macro2::Ident;
use syn::{ExprMethodCall, Type, PatType, GenericArgument, GenericParam, Generics};
use syn::punctuated::Punctuated;
use syn::visit_mut::{self, VisitMut};

pub struct FindReplaceExprMethodCall {
    pub find: Ident,
    pub replace: Ident,
}

pub struct FindReplaceGenericArgument {
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

impl VisitMut for FindReplaceExprMethodCall {
    fn visit_expr_method_call_mut(&mut self, node: &mut ExprMethodCall) {
        if node.method == self.find {
            node.method = self.replace.clone();
        }
        visit_mut::visit_expr_method_call_mut(self, node);
    }
}

impl VisitMut for FindReplaceGenericArgument {
    fn visit_generic_argument_mut(&mut self, node: &mut GenericArgument) {
        if let GenericArgument::Type(generic_type) = &node {
            if let Type::Path(type_path) = &generic_type {
                if type_path.path.is_ident(&self.find) {
                    *node = GenericArgument::Type(self.replace.clone())
                }
            }
        }

        visit_mut::visit_generic_argument_mut(self, node);
    }
}

impl VisitMut for RemoveGenerics {
    fn visit_generics_mut(&mut self, node: &mut Generics) {
        node.params = Punctuated::from_iter(node.params.iter().filter(|x| {
            if let GenericParam::Type(type_param) = &x {
                return type_param.ident != self.find;
            }

            true
        }).cloned());

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
