use proc_macro2::{Ident, TokenStream};
use syn::parse::{Parse, ParseStream};
use syn::token::Colon;
use syn::{ExprClosure, Result};
use quote::ToTokens;

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

impl ToTokens for ClosureDefinition {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.fn_name.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.closure.to_tokens(tokens);
    }
}
