use proc_macro2::Ident;
use std::ops::Deref;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::{As, Colon, Comma, Eq, Gt, In, Lt, Paren};
use syn::{Error, Result, Type, WherePredicate};

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

pub enum OperationBound {
    Type(WherePredicate),
    Eq(EqPredicate),
}

impl Parse for OperationBound {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek2(Eq) {
            Ok(OperationBound::Eq(input.parse()?))
        } else if input.peek2(Colon) {
            Ok(OperationBound::Type(input.parse()?))
        } else {
            Err(Error::new(input.span(), "Unexpected token."))
        }
    }
}

pub struct Operation {
    pub ident: Ident,
    pub lt_token: Option<Lt>,
    pub bound: Option<OperationBound>,
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
        let (lt_token, bound, gt_token) = match input.parse() {
            Ok(token) => {
                let lt_token = Some(token);
                let bound = Some(input.parse()?);
                let gt_token = Some(input.parse()?);

                (lt_token, bound, gt_token)
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
            bound,
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

impl Deref for OperationSequence {
    type Target = Punctuated<Operation, Comma>;
    fn deref(&self) -> &Self::Target {
        &self.sequence
    }
}
