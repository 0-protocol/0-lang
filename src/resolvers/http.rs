//! HTTP External Resolver
//!
//! Enables 0-lang graphs to make HTTP requests via External nodes.
//!
//! # URI Format
//!
//! ```text
//! http:{method}:{service}:{path}
//! ```
//!
//! - `method`: HTTP method (get, post, put, delete)
//! - `service`: Service identifier (maps to a base URL)
//! - `path`: Path and query string
//!
//! # Example URIs
//!
//! ```text
//! http:get:binance:/api/v3/ticker/price?symbol=BTCUSDT
//! http:post:binance:/api/v3/order
//! ```
//!
//! # Input Tensor Format
//!
//! For GET requests, inputs are ignored.
//! For POST/PUT requests, the first input tensor is serialized as the request body.
//!
//! # Output Tensor Format
//!
//! The response is converted to a tensor based on content type:
//! - JSON: Parsed and converted to tensor (numbers become data, confidence from HTTP status)
//! - Other: Raw bytes as tensor data

use std::collections::HashMap;
use std::sync::Arc;

use crate::{ExternalResolver, Tensor};

/// HTTP methods supported by the resolver
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
}

impl HttpMethod {
    /// Parse HTTP method from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "get" => Some(HttpMethod::Get),
            "post" => Some(HttpMethod::Post),
            "put" => Some(HttpMethod::Put),
            "delete" => Some(HttpMethod::Delete),
            _ => None,
        }
    }
}

/// HTTP External Resolver
///
/// Resolves HTTP requests from 0-lang graph External nodes.
pub struct HttpResolver {
    /// Base URLs for different services
    base_urls: HashMap<String, String>,
    /// Default headers to send with all requests
    #[allow(dead_code)]
    default_headers: HashMap<String, String>,
    /// Request timeout in milliseconds
    #[allow(dead_code)]
    timeout_ms: u64,
}

impl HttpResolver {
    /// Create a new HTTP resolver with default settings
    pub fn new() -> Self {
        Self {
            base_urls: HashMap::new(),
            default_headers: HashMap::new(),
            timeout_ms: 30_000, // 30 seconds default
        }
    }

    /// Create a builder for customizing the resolver
    pub fn builder() -> HttpResolverBuilder {
        HttpResolverBuilder::new()
    }

    /// Parse the URI format: "http:{method}:{service}:{path}"
    pub fn parse_uri(&self, uri: &str) -> Result<(HttpMethod, String, String), String> {
        let parts: Vec<&str> = uri.splitn(4, ':').collect();

        if parts.len() < 4 {
            return Err(format!(
                "Invalid HTTP URI format. Expected 'http:{{method}}:{{service}}:{{path}}', got: {}",
                uri
            ));
        }

        if parts[0] != "http" {
            return Err(format!("Expected 'http' prefix, got: {}", parts[0]));
        }

        let method = HttpMethod::from_str(parts[1])
            .ok_or_else(|| format!("Unknown HTTP method: {}", parts[1]))?;
        let service = parts[2].to_string();
        let path = parts[3].to_string();

        Ok((method, service, path))
    }

    /// Build the full URL from service and path
    pub fn build_url(&self, service: &str, path: &str) -> Result<String, String> {
        let base = self.base_urls.get(service).ok_or_else(|| {
            format!(
                "Unknown service: '{}'. Available services: {:?}",
                service,
                self.base_urls.keys().collect::<Vec<_>>()
            )
        })?;

        // Ensure path starts with /
        let path = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{}", path)
        };

        Ok(format!("{}{}", base, path))
    }

    /// Convert a tensor to JSON body for POST/PUT requests
    fn tensor_to_body(&self, tensor: &Tensor) -> String {
        // Simple conversion: tensor data as JSON array
        format!("{{\"data\":{:?},\"confidence\":{}}}", tensor.data, tensor.confidence)
    }

    /// Parse JSON response into a tensor
    #[allow(dead_code)]
    fn json_to_tensor(&self, json_str: &str, status_code: u16) -> Result<Tensor, String> {
        // Calculate confidence from HTTP status code
        let confidence = if status_code >= 200 && status_code < 300 {
            1.0
        } else if status_code >= 400 {
            0.0
        } else {
            0.5
        };

        // Try to parse as a number first
        if let Ok(val) = json_str.trim().parse::<f32>() {
            return Ok(Tensor::scalar(val, confidence));
        }

        // Try to parse as a JSON object and extract numeric values
        // This is a simple implementation - a full implementation would use serde_json
        let mut data: Vec<f32> = Vec::new();

        // Simple extraction of numbers from JSON (basic implementation)
        for part in json_str.split([',', ':', '{', '}', '[', ']', '"']) {
            let trimmed = part.trim();
            if let Ok(val) = trimmed.parse::<f32>() {
                data.push(val);
            }
        }

        if data.is_empty() {
            // Return a scalar 1.0 to indicate successful response
            Ok(Tensor::scalar(1.0, confidence))
        } else {
            Ok(Tensor::new(vec![data.len() as u32], data, confidence))
        }
    }
}

impl Default for HttpResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ExternalResolver for HttpResolver {
    fn resolve(&self, uri: &str, inputs: Vec<&Tensor>) -> Result<Tensor, String> {
        let (method, service, path) = self.parse_uri(uri)?;
        let _url = self.build_url(&service, &path)?;

        // NOTE: This is a synchronous placeholder implementation.
        // A real implementation would use an async HTTP client like reqwest
        // integrated with a tokio runtime.
        //
        // For now, we return a placeholder tensor indicating the request
        // was parsed successfully. The actual HTTP call would be made
        // when integrating with 0-hummingbot or another application.

        // Simulate a successful response
        match method {
            HttpMethod::Get => {
                // GET requests return placeholder data
                Ok(Tensor::scalar(1.0, 0.5))
            }
            HttpMethod::Post | HttpMethod::Put => {
                // POST/PUT use the first input as body
                if let Some(input) = inputs.first() {
                    let _body = self.tensor_to_body(input);
                    Ok(Tensor::scalar(1.0, 0.5))
                } else {
                    Ok(Tensor::scalar(0.0, 0.5))
                }
            }
            HttpMethod::Delete => {
                // DELETE returns success indicator
                Ok(Tensor::scalar(1.0, 0.5))
            }
        }
    }
}

/// Builder for HttpResolver
pub struct HttpResolverBuilder {
    base_urls: HashMap<String, String>,
    default_headers: HashMap<String, String>,
    timeout_ms: u64,
}

impl HttpResolverBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            base_urls: HashMap::new(),
            default_headers: HashMap::new(),
            timeout_ms: 30_000,
        }
    }

    /// Add a base URL for a service
    pub fn with_service(mut self, name: &str, base_url: &str) -> Self {
        self.base_urls.insert(name.to_string(), base_url.to_string());
        self
    }

    /// Add common exchange endpoints
    pub fn with_exchanges(self) -> Self {
        self.with_service("binance", "https://api.binance.com")
            .with_service("binance-futures", "https://fapi.binance.com")
            .with_service("okx", "https://www.okx.com")
            .with_service("hyperliquid", "https://api.hyperliquid.xyz")
            .with_service("kucoin", "https://api.kucoin.com")
            .with_service("gate", "https://api.gateio.ws")
    }

    /// Add a default header
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.default_headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Set the request timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Build the resolver
    pub fn build(self) -> HttpResolver {
        HttpResolver {
            base_urls: self.base_urls,
            default_headers: self.default_headers,
            timeout_ms: self.timeout_ms,
        }
    }

    /// Build as Arc<dyn ExternalResolver>
    pub fn build_arc(self) -> Arc<dyn ExternalResolver> {
        Arc::new(self.build())
    }
}

impl Default for HttpResolverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_uri() {
        let resolver = HttpResolver::builder()
            .with_service("binance", "https://api.binance.com")
            .build();

        let (method, service, path) = resolver
            .parse_uri("http:get:binance:/api/v3/ticker/price?symbol=BTCUSDT")
            .unwrap();

        assert_eq!(method, HttpMethod::Get);
        assert_eq!(service, "binance");
        assert_eq!(path, "/api/v3/ticker/price?symbol=BTCUSDT");
    }

    #[test]
    fn test_parse_post_uri() {
        let resolver = HttpResolver::builder()
            .with_service("binance", "https://api.binance.com")
            .build();

        let (method, service, path) = resolver
            .parse_uri("http:post:binance:/api/v3/order")
            .unwrap();

        assert_eq!(method, HttpMethod::Post);
        assert_eq!(service, "binance");
        assert_eq!(path, "/api/v3/order");
    }

    #[test]
    fn test_build_url() {
        let resolver = HttpResolver::builder()
            .with_service("binance", "https://api.binance.com")
            .build();

        let url = resolver.build_url("binance", "/api/v3/ticker").unwrap();
        assert_eq!(url, "https://api.binance.com/api/v3/ticker");
    }

    #[test]
    fn test_unknown_service() {
        let resolver = HttpResolver::new();
        let result = resolver.build_url("unknown", "/path");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_uri() {
        let resolver = HttpResolver::new();
        let result = resolver.parse_uri("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_exchanges() {
        let resolver = HttpResolver::builder()
            .with_exchanges()
            .build();

        assert!(resolver.base_urls.contains_key("binance"));
        assert!(resolver.base_urls.contains_key("okx"));
        assert!(resolver.base_urls.contains_key("hyperliquid"));
    }

    #[test]
    fn test_resolve_get() {
        let resolver = HttpResolver::builder()
            .with_service("test", "https://test.com")
            .build();

        let result = resolver.resolve("http:get:test:/api/data", vec![]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_post() {
        let resolver = HttpResolver::builder()
            .with_service("test", "https://test.com")
            .build();

        let input = Tensor::scalar(42.0, 1.0);
        let result = resolver.resolve("http:post:test:/api/data", vec![&input]);
        assert!(result.is_ok());
    }
}
