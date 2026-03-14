//! JSON Operations for ZeroLang
//!
//! Provides operations for parsing and extracting data from JSON responses,
//! commonly needed for interacting with trading APIs.

use crate::tensor::{Tensor, TensorData};
use rust_decimal::Decimal;
use serde_json::Value;

/// Errors that can occur during JSON operations
#[derive(Debug, Clone)]
pub enum JsonError {
    /// Failed to parse JSON string
    ParseError(String),
    /// Key path not found in JSON
    KeyNotFound(String),
    /// Type mismatch when extracting value
    TypeMismatch { expected: String, got: String },
    /// Invalid array index
    IndexOutOfBounds { index: usize, len: usize },
    /// Input tensor is not a string tensor
    NotStringTensor,
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonError::ParseError(e) => write!(f, "JSON parse error: {}", e),
            JsonError::KeyNotFound(key) => write!(f, "Key not found: {}", key),
            JsonError::TypeMismatch { expected, got } => {
                write!(f, "Type mismatch: expected {}, got {}", expected, got)
            }
            JsonError::IndexOutOfBounds { index, len } => {
                write!(f, "Array index {} out of bounds (len: {})", index, len)
            }
            JsonError::NotStringTensor => write!(f, "Input tensor is not a string tensor"),
        }
    }
}

impl std::error::Error for JsonError {}

/// Parse a JSON string tensor into a structured representation.
/// Returns a StringTensor containing the parsed JSON as a serialized form.
pub fn json_parse(input: &Tensor) -> Result<Tensor, JsonError> {
    let json_string = match &input.data {
        TensorData::String(v) => {
            if v.is_empty() {
                return Err(JsonError::ParseError("Empty string tensor".to_string()));
            }
            &v[0]
        }
        _ => return Err(JsonError::NotStringTensor),
    };

    // Parse the JSON to validate it
    let _: Value = serde_json::from_str(json_string)
        .map_err(|e| JsonError::ParseError(e.to_string()))?;

    // Return the original JSON string as a parsed tensor (validation succeeded)
    // In a full implementation, we might transform this into a structured format
    Ok(Tensor {
        shape: vec![1],
        data: TensorData::String(vec![json_string.clone()]),
        confidence: input.confidence,
    })
}

/// Extract a value from JSON by key path (e.g., "data.price" or "result.trades[0].amount")
pub fn json_get(json_tensor: &Tensor, key_path: &str) -> Result<Tensor, JsonError> {
    let json_string = match &json_tensor.data {
        TensorData::String(v) => {
            if v.is_empty() {
                return Err(JsonError::ParseError("Empty string tensor".to_string()));
            }
            &v[0]
        }
        _ => return Err(JsonError::NotStringTensor),
    };

    let value: Value = serde_json::from_str(json_string)
        .map_err(|e| JsonError::ParseError(e.to_string()))?;

    let extracted = get_value_by_path(&value, key_path)?;
    value_to_tensor(extracted, json_tensor.confidence)
}

/// Extract array elements from a JSON array
pub fn json_array(json_tensor: &Tensor) -> Result<Tensor, JsonError> {
    let json_string = match &json_tensor.data {
        TensorData::String(v) => {
            if v.is_empty() {
                return Err(JsonError::ParseError("Empty string tensor".to_string()));
            }
            &v[0]
        }
        _ => return Err(JsonError::NotStringTensor),
    };

    let value: Value = serde_json::from_str(json_string)
        .map_err(|e| JsonError::ParseError(e.to_string()))?;

    match value {
        Value::Array(arr) => {
            // Convert each array element to a string representation
            let strings: Vec<String> = arr
                .iter()
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .collect();
            
            let len = strings.len() as u32;
            Ok(Tensor {
                shape: vec![len],
                data: TensorData::String(strings),
                confidence: json_tensor.confidence,
            })
        }
        _ => Err(JsonError::TypeMismatch {
            expected: "array".to_string(),
            got: json_type_name(&value),
        }),
    }
}

/// Navigate through JSON using a dot-notation path
fn get_value_by_path<'a>(value: &'a Value, path: &str) -> Result<&'a Value, JsonError> {
    let mut current = value;
    
    for segment in path.split('.') {
        // Check for array index notation like "trades[0]"
        if let Some(bracket_pos) = segment.find('[') {
            let key = &segment[..bracket_pos];
            let index_str = &segment[bracket_pos + 1..segment.len() - 1];
            
            // First, navigate to the key
            if !key.is_empty() {
                current = current
                    .get(key)
                    .ok_or_else(|| JsonError::KeyNotFound(key.to_string()))?;
            }
            
            // Then, navigate to the array index
            let index: usize = index_str
                .parse()
                .map_err(|_| JsonError::ParseError(format!("Invalid array index: {}", index_str)))?;
            
            let arr = current
                .as_array()
                .ok_or_else(|| JsonError::TypeMismatch {
                    expected: "array".to_string(),
                    got: json_type_name(current),
                })?;
            
            current = arr
                .get(index)
                .ok_or_else(|| JsonError::IndexOutOfBounds { index, len: arr.len() })?;
        } else if !segment.is_empty() {
            // Simple key navigation
            current = current
                .get(segment)
                .ok_or_else(|| JsonError::KeyNotFound(segment.to_string()))?;
        }
    }
    
    Ok(current)
}

/// Convert a JSON value to a Tensor
fn value_to_tensor(value: &Value, confidence: f32) -> Result<Tensor, JsonError> {
    match value {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                Ok(Tensor::scalar(f as f32, confidence))
            } else if let Some(i) = n.as_i64() {
                Ok(Tensor::scalar(i as f32, confidence))
            } else {
                // Try to parse as Decimal for high-precision numbers
                let decimal_str = n.to_string();
                if let Ok(d) = decimal_str.parse::<Decimal>() {
                    Ok(Tensor::scalar_decimal(d, confidence))
                } else {
                    Err(JsonError::TypeMismatch {
                        expected: "number".to_string(),
                        got: "unparseable number".to_string(),
                    })
                }
            }
        }
        Value::String(s) => Ok(Tensor::scalar_string(s.clone(), confidence)),
        Value::Bool(b) => Ok(Tensor::scalar(if *b { 1.0 } else { 0.0 }, confidence)),
        Value::Array(arr) => {
            // Try to convert to float array if all elements are numbers
            let floats: Result<Vec<f32>, _> = arr
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| JsonError::TypeMismatch {
                            expected: "number".to_string(),
                            got: json_type_name(v),
                        })
                })
                .collect();
            
            match floats {
                Ok(f) => Ok(Tensor::from_vec(f, confidence)),
                Err(_) => {
                    // Fall back to string array
                    let strings: Vec<String> = arr
                        .iter()
                        .map(|v| serde_json::to_string(v).unwrap_or_default())
                        .collect();
                    Ok(Tensor::from_strings(strings, confidence))
                }
            }
        }
        Value::Object(_) => {
            // Serialize object back to JSON string
            let s = serde_json::to_string(value)
                .map_err(|e| JsonError::ParseError(e.to_string()))?;
            Ok(Tensor::scalar_string(s, confidence))
        }
        Value::Null => Ok(Tensor::scalar(f32::NAN, confidence * 0.5)), // Null reduces confidence
    }
}

/// Get the type name of a JSON value
fn json_type_name(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "bool".to_string(),
        Value::Number(_) => "number".to_string(),
        Value::String(_) => "string".to_string(),
        Value::Array(_) => "array".to_string(),
        Value::Object(_) => "object".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_parse() {
        let json = r#"{"price": 42000.50, "symbol": "BTC/USD"}"#;
        let tensor = Tensor::scalar_string(json.to_string(), 1.0);
        let result = json_parse(&tensor).unwrap();
        assert!(result.data.is_string());
    }

    #[test]
    fn test_json_get_number() {
        let json = r#"{"data": {"price": 42000.50}}"#;
        let tensor = Tensor::scalar_string(json.to_string(), 1.0);
        let result = json_get(&tensor, "data.price").unwrap();
        assert!((result.as_scalar() - 42000.50).abs() < 0.01);
    }

    #[test]
    fn test_json_get_string() {
        let json = r#"{"symbol": "BTC/USD"}"#;
        let tensor = Tensor::scalar_string(json.to_string(), 1.0);
        let result = json_get(&tensor, "symbol").unwrap();
        assert_eq!(result.as_scalar_string(), "BTC/USD");
    }

    #[test]
    fn test_json_get_array_element() {
        let json = r#"{"trades": [{"price": 100}, {"price": 200}]}"#;
        let tensor = Tensor::scalar_string(json.to_string(), 1.0);
        let result = json_get(&tensor, "trades[1].price").unwrap();
        assert_eq!(result.as_scalar(), 200.0);
    }

    #[test]
    fn test_json_array() {
        let json = r#"["BTC/USD", "ETH/USD", "SOL/USD"]"#;
        let tensor = Tensor::scalar_string(json.to_string(), 1.0);
        let result = json_array(&tensor).unwrap();
        assert_eq!(result.shape, vec![3]);
        let strings = result.data.as_string().unwrap();
        assert_eq!(strings[0], "\"BTC/USD\"");
        assert_eq!(strings[1], "\"ETH/USD\"");
    }

    #[test]
    fn test_json_get_key_not_found() {
        let json = r#"{"price": 100}"#;
        let tensor = Tensor::scalar_string(json.to_string(), 1.0);
        let result = json_get(&tensor, "nonexistent");
        assert!(matches!(result, Err(JsonError::KeyNotFound(_))));
    }

    #[test]
    fn test_json_parse_error() {
        let invalid_json = r#"{"price": invalid}"#;
        let tensor = Tensor::scalar_string(invalid_json.to_string(), 1.0);
        let result = json_parse(&tensor);
        assert!(matches!(result, Err(JsonError::ParseError(_))));
    }
}
