//! External Resolvers for ZeroLang
//!
//! This module provides external resolver implementations for common
//! use cases like HTTP requests, JSON parsing, and more.

pub mod http;

pub use http::{HttpMethod, HttpResolver, HttpResolverBuilder};
