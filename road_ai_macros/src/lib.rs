extern crate proc_macro;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};

use syn::{ItemImpl, ImplItem, WhereClause, parse_macro_input};
use syn::token::{Where};
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use quote::quote;

mod operation_syntax;
use operation_syntax::*;

mod search_replace;
use search_replace::*;

#[proc_macro_attribute]
pub fn expand_operations(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse input and impl block.
    let operation_sequence = parse_macro_input!(attr as OperationSequence);
    let item = parse_macro_input!(item as ItemImpl);

    let mut impl_blocks = Vec::new();

    for operation in operation_sequence.iter() {
        let name = match &operation.alias {
            Some(alias) => alias,
            None => &operation.ident,
        };
        
        let mut impl_block = item.clone();
        let mut placeholder_method_visitor = FindReplaceExprMethodCall {
            find: Ident::new("placeholder", Span::call_site()),
            replace: operation.ident.clone(),
        };
        let mut unchecked_method_visitor = FindReplaceExprMethodCall {
            find: Ident::new("unchecked", Span::call_site()),
            replace: Ident::new(&format!("{}_unchecked", name), Span::call_site()),
        };

        // Add trait bound if specified by the operation.
        // Marginalize generic if an equality bound is specified instead.
        if let Some(bound) = &operation.bound {
            match bound {
                OperationBound::Type(trait_bound) => {
                    impl_block.generics.where_clause = if let Some(mut where_clause) = impl_block.generics.where_clause {
                        where_clause.predicates.push(trait_bound.clone());
                        Some(where_clause)
                    } else {
                        let mut predicates = Punctuated::new();
                        predicates.push(trait_bound.clone());
        
                        Some(WhereClause {
                            where_token: Where::default(),
                            predicates,
                        })
                    }
                },
                OperationBound::Eq(margin) => {
                    let mut generics_visitor = RemoveGenerics {
                        find: margin.lhs_ty.clone(),
                    };
                    let mut generic_argument_visitor = FindReplaceGenericArgument {
                        find: margin.lhs_ty.clone(),
                        replace: margin.rhs_ty.clone(),
                    };
            
                    generics_visitor.visit_item_impl_mut(&mut impl_block);
                    generic_argument_visitor.visit_item_impl_mut(&mut impl_block);
                }
            }
        }

        // Replace placeholder types with given types.
        if let Some(types) = &operation.types {
            for (i, t) in types.iter().enumerate() {                
                let mut pat_type_visitor = FindReplacePatType {
                    find: Ident::new(&format!("type{}", i), Span::call_site()),
                    replace: t.clone()
                };

                pat_type_visitor.visit_item_impl_mut(&mut impl_block);
            }
        }
        
        // Adapt all methods to the operation.
        for impl_item in impl_block.items.iter_mut() {
            if let ImplItem::Method(method) = impl_item {
                // Prepend the operation name to the method's name.
                method.sig.ident = if method.sig.ident == Ident::new("operation", Span::call_site()) {
                    Ident::new(&format!("{}", name), Span::call_site())
                } else {
                    Ident::new(&format!("{}_{}", name, method.sig.ident), Span::call_site())
                };

                placeholder_method_visitor.visit_block_mut(&mut method.block);
                unchecked_method_visitor.visit_block_mut(&mut method.block);
            }
        }

        impl_blocks.push(impl_block);
    }

    let result = quote! {
        #(#impl_blocks)*
    };
    result.into()
}
