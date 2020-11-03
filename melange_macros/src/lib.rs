extern crate proc_macro;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};

use quote::quote;
use syn::punctuated::Punctuated;
use syn::token::Where;
use syn::visit_mut::VisitMut;
use syn::{parse_macro_input, ImplItem, ItemImpl, ItemTrait, TraitItem, Result, WhereClause, Type};

mod operation_syntax;
use operation_syntax::*;

mod closure_definition_syntax;
use closure_definition_syntax::*;

mod search_replace;
use search_replace::*;

#[proc_macro_attribute]
pub fn expand_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse input and impl block.
    let operation_sequence = parse_macro_input!(attr as OperationSequence);
    let item = parse_macro_input!(item as ItemImpl);

    let mut impl_blocks = Vec::new();

    // Apply changes corresponding to each operation in the sequence.
    for operation in operation_sequence.iter() {
        let name = match &operation.alias {
            Some(alias) => alias,
            None => &operation.ident,
        };
        let mut impl_block = item.clone();

        // Turn impl to trait impl if needed
        if let Some(trait_impl) = &operation.trait_impl {
            if let Some(trait_) = &mut impl_block.trait_ {
                if let Some(trait_path) = trait_.1.segments.last_mut() {
                    trait_path.ident = trait_impl.clone();
                }
            }
        }

        // Visitor that replaces calls to `placeholer` by the operation's name.
        let mut placeholder_method_visitor = FindReplaceExprMethodCall {
            find: Ident::new("placeholder", Span::call_site()),
            replace: operation.ident.clone(),
        };

        // Visitor that replaces calls to `unchecked` by the operation's unchecked method.
        let mut unchecked_method_visitor = FindReplaceExprMethodCall {
            find: Ident::new("unchecked", Span::call_site()),
            replace: Ident::new(&format!("{}_unchecked", name), Span::call_site()),
        };
        let mut unchecked_underscore_method_visitor = FindReplaceExprMethodCall {
            find: Ident::new("unchecked_", Span::call_site()),
            replace: Ident::new(&format!("{}_unchecked_", name), Span::call_site()),
        };

        // For all bounds:
        // - Add trait bound if specified by the operation.
        // - Marginalize generic if an equality bound is specified instead.
        if let Some(bounds) = &operation.bounds {
            for bound in bounds.iter() {
                match bound {
                    OperationBound::Type(trait_bound) => {
                        impl_block.generics.where_clause =
                            if let Some(mut where_clause) = impl_block.generics.where_clause {
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
                    }
                    OperationBound::Eq(margin) => {
                        let mut generics_visitor = RemoveGenerics {
                            find: margin.lhs_ty.clone(),
                        };
                        let mut type_visitor = FindReplaceType {
                            find: margin.lhs_ty.clone(),
                            replace: margin.rhs_ty.clone(),
                        };
                        generics_visitor.visit_item_impl_mut(&mut impl_block);
                        type_visitor.visit_item_impl_mut(&mut impl_block);

                        if let Type::Path(path) = &margin.rhs_ty {
                            if let Some(_) = path.path.get_ident() {
                                let rhs_ty = &margin.rhs_ty;
                                let mut visitor = FindReplaceIdent {
                                    find: margin.lhs_ty.clone(),
                                    replace: Ident::new(
                                        &format!("{}", quote! { #rhs_ty }),
                                        Span::call_site(),
                                    ),
                                };

                                for attribute in impl_block.attrs.iter_mut() {
                                    let parsed_attribute: Result<ClosureDefinition> = attribute.parse_args();
                                    match parsed_attribute {
                                        Ok(mut def) => {
                                            // let fn_name = def.fn_name;
                                            // let mut closure = def.closure;
                                            // let rhs_ty = &margin.rhs_ty;
                                            // let mut visitor = FindReplaceIdent {
                                            //     find: margin.lhs_ty.clone(),
                                            //     replace: Ident::new(
                                            //         &format!("{}", quote! { #rhs_ty }),
                                            //         Span::call_site(),
                                            //     ),
                                            // };
                                            visitor.visit_expr_closure_mut(&mut def.closure);
            
                                            attribute.tokens = quote! { (#def) };
                                        },
                                        Err(_) => {
                                            let parsed_attribute: Result<OperationSequence> = attribute.parse_args();
                                            match parsed_attribute {
                                                Ok(mut seq) => {
                                                    for operation in seq.iter_mut() {
                                                        if let Some(bounds) = &mut operation.bounds {
                                                            for bound in bounds.iter_mut() {
                                                                match bound {
                                                                    OperationBound::Type(trait_bound) => {
                                                                        visitor.visit_where_predicate_mut(trait_bound);
                                                                    }
                                                                    OperationBound::Eq(margin) => {
                                                                        visitor.visit_type_mut(&mut margin.rhs_ty);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    attribute.tokens = quote! { (#seq) };
                                                },
                                                Err(_) => {},
                                            }
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Replace placeholder types with given types.
        if let Some(types) = &operation.types {
            for (i, t) in types.iter().enumerate() {
                let mut pat_type_visitor = FindReplacePatType {
                    find: Ident::new(&format!("type{}", i), Span::call_site()),
                    replace: t.clone(),
                };

                pat_type_visitor.visit_item_impl_mut(&mut impl_block);
            }
        }
        // Adapt all methods to the operation.
        for impl_item in impl_block.items.iter_mut() {
            if let ImplItem::Method(method) = impl_item {
                // Prepend the operation name to the method's name.
                // If the method's name constains operation, replace it with the operation name.
                let mut method_name = format!("{}", method.sig.ident);
                if let Some(position) = method_name.find("operation") {
                    method_name.replace_range(position..position + 9, &format!("{}", name));
                    method.sig.ident = Ident::new(&method_name, Span::call_site())
                };

                placeholder_method_visitor.visit_block_mut(&mut method.block);
                unchecked_method_visitor.visit_block_mut(&mut method.block);
                unchecked_underscore_method_visitor.visit_block_mut(&mut method.block);
            }
        }

        impl_blocks.push(impl_block);
    }

    let result = quote! {
        #(#impl_blocks)*
    };
    result.into()
}

#[proc_macro_attribute]
pub fn define_closure(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as ClosureDefinition);
    let mut item = parse_macro_input!(item as ItemImpl);

    for impl_item in item.items.iter_mut() {
        if let ImplItem::Method(method) = impl_item {
            if attr.fn_name == method.sig.ident {
                let mut visitor = ReplaceExprClosure {
                    replace: attr.closure,
                };

                visitor.visit_block_mut(&mut method.block);
                break;
            }
        }
    }

    let result = quote! {
        #item
    };
    result.into()
}

#[proc_macro_attribute]
pub fn expand_trait(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as IdentSequence);
    let item = parse_macro_input!(item as ItemTrait);

    let mut trait_blocks = Vec::new();

    for ident in attr.iter() {
        let mut trait_block = item.clone();
        trait_block.ident = ident.clone();

        let trait_name = format!("{}", ident);
        let mut method_name = String::new();
        for c in trait_name.chars() {
            if c.is_ascii_uppercase() {
                method_name.push('_');
                method_name.push(c.to_ascii_lowercase());
            } else {
                method_name.push(c);
            }
        }
        let method_name = &method_name[1..];

        for trait_item in trait_block.items.iter_mut() {
            if let TraitItem::Method(method) = trait_item {
                if &format!("{}", method.sig.ident) == "operation" {
                    method.sig.ident = Ident::new(method_name, Span::call_site());
                }
            }
        }

        trait_blocks.push(trait_block);
    }

    let result = quote! {
        #(#trait_blocks)*
    };
    result.into()
}
