use proc_macro2::{Ident, TokenStream, Spacing, Punct};
use std::ops::{Deref, DerefMut};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::{As, Comma, Eq, Gt, In, Lt, Paren};
use syn::{Result, Type, WherePredicate, punctuated::Pair};
use quote::{ToTokens, TokenStreamExt};

pub struct EqPredicate {
    pub lhs_ty: Ident,
    pub eq_token: Eq,
    pub rhs_ty: Type,
}

impl Parse for EqPredicate {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(EqPredicate {
            lhs_ty: input.parse()?,
            eq_token: input.parse()?,
            rhs_ty: input.parse()?,
        })
    }
}

impl ToTokens for EqPredicate {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.lhs_ty.to_tokens(tokens);
        self.eq_token.to_tokens(tokens);
        self.rhs_ty.to_tokens(tokens);
    }
}

pub enum OperationBound {
    Type(WherePredicate),
    Eq(EqPredicate),
}

impl Parse for OperationBound {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek2(Eq) {
            Ok(OperationBound::Eq(input.parse()?))
        // } else if input.peek2(Colon) {
        //     Ok(OperationBound::Type(input.parse()?))
        // } else {
        //     Err(Error::new(input.span(), "Unexpected token."))
        // }
        } else {
            Ok(OperationBound::Type(input.parse()?))
        }
    }
}

impl ToTokens for OperationBound {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Type(predicate) => predicate.to_tokens(tokens),
            Self::Eq(predicate) => predicate.to_tokens(tokens),
        }
    }
}

pub struct Operation {
    pub ident: Ident,
    pub lt_token: Option<Lt>,
    pub bounds: Option<Punctuated<OperationBound, Comma>>,
    pub gt_token: Option<Gt>,
    pub paren_token: Option<Paren>,
    pub types: Option<Punctuated<Type, Comma>>,
    pub as_token: Option<As>,
    pub alias: Option<Ident>,
    pub in_token: Option<In>,
    pub trait_impl: Option<Ident>,
}

impl Parse for Operation {
    fn parse(input: ParseStream) -> Result<Self> {
        let ident = input.parse()?;
        let (lt_token, bounds, gt_token) = match input.parse() {
            Ok(token) => {
                let lt_token = Some(token);
                let bounds = Some({
                    let mut args = Punctuated::new();
                    loop {
                        if input.peek(Gt) {
                            break;
                        }
                        let value = input.parse()?;
                        args.push_value(value);
                        if input.peek(Gt) {
                            break;
                        }
                        let punct = input.parse()?;
                        args.push_punct(punct);
                    }
                    args
                });
                let gt_token = Some(input.parse()?);

                (lt_token, bounds, gt_token)
            }
            Err(_) => (None, None, None),
        };

        let (paren_token, types) = match syn::group::parse_parens(&input) {
            Ok(parens) => (
                Some(parens.token),
                Some(Punctuated::parse_terminated(&parens.content)?),
            ),
            Err(_) => (None, None),
        };

        let (as_token, alias) = match input.parse() {
            Ok(token) => {
                let as_token = Some(token);
                let alias = Some(input.parse()?);

                (as_token, alias)
            }
            Err(_) => (None, None),
        };

        let (in_token, trait_impl) = match input.parse() {
            Ok(token) => {
                let in_token = Some(token);
                let trait_impl = Some(input.parse()?);

                (in_token, trait_impl)
            }
            Err(_) => (None, None),
        };
        Ok(Operation {
            ident,
            lt_token,
            bounds,
            gt_token,
            paren_token,
            types,
            as_token,
            alias,
            in_token,
            trait_impl,
        })
    }
}

impl ToTokens for Operation {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.ident.to_tokens(tokens);

        if let Some(lt_token) = &self.lt_token {
            lt_token.to_tokens(tokens);
            for pair in self.bounds.as_ref().unwrap().pairs() {
                match pair {
                    Pair::Punctuated(bound, comma) => {
                        bound.to_tokens(tokens);
                        comma.to_tokens(tokens);
                    },
                    Pair::End(bound) => {
                        bound.to_tokens(tokens);
                    }
                }
            }
            self.gt_token.as_ref().unwrap().to_tokens(tokens);
        }
        
        
        if let Some(_) = &self.paren_token {
            tokens.append(Punct::new('(', Spacing::Alone));
            for pair in self.types.as_ref().unwrap().pairs() {
                match pair {
                    Pair::Punctuated(ty, comma) => {
                        ty.to_tokens(tokens);
                        comma.to_tokens(tokens);
                    },
                    Pair::End(ty) => {
                        ty.to_tokens(tokens);
                    }
                }
            }
            tokens.append(Punct::new(')', Spacing::Alone));
        }

        if let Some(as_token) = &self.as_token {
            as_token.to_tokens(tokens);
            self.alias.as_ref().unwrap().to_tokens(tokens)
        }

        if let Some(in_token) = &self.in_token {
            in_token.to_tokens(tokens);
            self.trait_impl.as_ref().unwrap().to_tokens(tokens)
        }
    }
}

pub struct OperationSequence {
    pub sequence: Punctuated<Operation, Comma>,
}

impl Parse for OperationSequence {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(OperationSequence {
            sequence: Punctuated::parse_terminated(input)?,
        })
    }
}

impl ToTokens for OperationSequence {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for pair in self.sequence.pairs() {
            match pair {
                Pair::Punctuated(operation, comma) => {
                    operation.to_tokens(tokens);
                    comma.to_tokens(tokens);
                },
                Pair::End(operation) => {
                    operation.to_tokens(tokens);
                }
            }
        }
    }
}

impl Deref for OperationSequence {
    type Target = Punctuated<Operation, Comma>;
    fn deref(&self) -> &Self::Target {
        &self.sequence
    }
}

impl DerefMut for OperationSequence {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.sequence
    }
}

pub struct IdentSequence {
    pub sequence: Punctuated<Ident, Comma>,
}

impl Parse for IdentSequence {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(IdentSequence {
            sequence: Punctuated::parse_terminated(input)?,
        })
    }
}

impl ToTokens for IdentSequence {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for pair in self.sequence.pairs() {
            match pair {
                Pair::Punctuated(ident, comma) => {
                    ident.to_tokens(tokens);
                    comma.to_tokens(tokens);
                },
                Pair::End(ident) => {
                    ident.to_tokens(tokens);
                }
            }
        }
    }
}

impl Deref for IdentSequence {
    type Target = Punctuated<Ident, Comma>;
    fn deref(&self) -> &Self::Target {
        &self.sequence
    }
}

impl DerefMut for IdentSequence {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.sequence
    }
}
