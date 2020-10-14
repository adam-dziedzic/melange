use proc_macro2::Ident;
use syn::parse::{Parse, ParseStream};
use syn::token::Colon;
use syn::{ExprClosure, Result};

pub struct ClosureDefinition {
    pub fn_name: Ident,
    pub colon_token: Colon,
    pub closure: ExprClosure,
}

impl Parse for ClosureDefinition {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ClosureDefinition {
            fn_name: input.parse()?,
            colon_token: input.parse()?,
            closure: input.parse()?,
        })
    }
}
