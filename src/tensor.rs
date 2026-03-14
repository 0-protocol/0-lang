//! Tensor - The fundamental data type in ZeroLang
//!
//! Replaces int, float, bool, string with probabilistic vectors.
//! Every value carries a confidence score.

use rust_decimal::Decimal;
use std::collections::VecDeque;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Handle to a streaming data source
#[derive(Debug, Clone)]
pub struct StreamHandle {
    /// Unique identifier for this stream
    pub id: u64,
    /// Type of stream source
    pub source_type: StreamSource,
    /// Buffer for incoming tensors
    pub buffer: Arc<RwLock<VecDeque<Tensor>>>,
}

impl PartialEq for StreamHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.source_type == other.source_type
    }
}

/// Source type for streaming data
#[derive(Debug, Clone, PartialEq)]
pub enum StreamSource {
    /// WebSocket connection
    WebSocket { url: String },
    /// Internal channel
    Channel { channel_id: String },
    /// Event stream (SSE)
    Event { event_type: String },
}

/// The data stored in a tensor - supports multiple types for trading applications.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    /// Standard floating point data (original type)
    Float(Vec<f32>),
    /// String data for API responses, trading pairs, etc.
    String(Vec<String>),
    /// Decimal data for financial precision calculations
    Decimal(Vec<Decimal>),
    /// Streaming data source (WebSocket, SSE, channels)
    Stream(StreamHandle),
}

impl TensorData {
    /// Get the number of elements in the tensor data
    pub fn len(&self) -> usize {
        match self {
            TensorData::Float(v) => v.len(),
            TensorData::String(v) => v.len(),
            TensorData::Decimal(v) => v.len(),
            TensorData::Stream(_) => 0, // Streams have dynamic length
        }
    }

    /// Check if the tensor data is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if this is float data
    pub fn is_float(&self) -> bool {
        matches!(self, TensorData::Float(_))
    }

    /// Check if this is string data
    pub fn is_string(&self) -> bool {
        matches!(self, TensorData::String(_))
    }

    /// Check if this is decimal data
    pub fn is_decimal(&self) -> bool {
        matches!(self, TensorData::Decimal(_))
    }

    /// Check if this is stream data
    pub fn is_stream(&self) -> bool {
        matches!(self, TensorData::Stream(_))
    }

    /// Get as float data (returns None if not float type)
    pub fn as_float(&self) -> Option<&Vec<f32>> {
        match self {
            TensorData::Float(v) => Some(v),
            _ => None,
        }
    }

    /// Get as string data (returns None if not string type)
    pub fn as_string(&self) -> Option<&Vec<String>> {
        match self {
            TensorData::String(v) => Some(v),
            _ => None,
        }
    }

    /// Get as decimal data (returns None if not decimal type)
    pub fn as_decimal(&self) -> Option<&Vec<Decimal>> {
        match self {
            TensorData::Decimal(v) => Some(v),
            _ => None,
        }
    }

    /// Get as stream handle (returns None if not stream type)
    pub fn as_stream(&self) -> Option<&StreamHandle> {
        match self {
            TensorData::Stream(h) => Some(h),
            _ => None,
        }
    }
}

/// The fundamental data type in ZeroLang.
///
/// In traditional languages, you have `int`, `float`, `bool`, `string`.
/// In ZeroLang, everything is a Tensor with a confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Dimensions of the tensor, e.g., [768] for embedding, [1] for scalar
    pub shape: Vec<u32>,
    /// Flattened tensor data (supports multiple types)
    pub data: TensorData,
    /// Meta-confidence in this tensor's validity [0.0, 1.0]
    /// This is the "Schrödinger" probability - how certain are we about this value?
    pub confidence: f32,
}

/// Legacy accessor for backwards compatibility - returns float data reference
impl Tensor {
    /// Get float data directly (for backwards compatibility)
    /// Panics if the tensor contains non-float data
    pub fn float_data(&self) -> &Vec<f32> {
        self.data.as_float().expect("Tensor does not contain float data")
    }

    /// Get mutable float data directly (for backwards compatibility)
    /// Panics if the tensor contains non-float data
    pub fn float_data_mut(&mut self) -> &mut Vec<f32> {
        match &mut self.data {
            TensorData::Float(v) => v,
            _ => panic!("Tensor does not contain float data"),
        }
    }
}

impl Tensor {
    /// Create a new tensor with given shape, float data, and confidence
    pub fn new(shape: Vec<u32>, data: Vec<f32>, confidence: f32) -> Self {
        Self {
            shape,
            data: TensorData::Float(data),
            confidence,
        }
    }

    /// Create a new tensor with typed data
    pub fn with_data(shape: Vec<u32>, data: TensorData, confidence: f32) -> Self {
        Self {
            shape,
            data,
            confidence,
        }
    }

    /// Create a scalar tensor (shape [1]) with a single value
    pub fn scalar(value: f32, confidence: f32) -> Self {
        Self {
            shape: vec![1],
            data: TensorData::Float(vec![value]),
            confidence,
        }
    }

    /// Create a scalar tensor with a decimal value
    pub fn scalar_decimal(value: Decimal, confidence: f32) -> Self {
        Self {
            shape: vec![1],
            data: TensorData::Decimal(vec![value]),
            confidence,
        }
    }

    /// Create a scalar tensor with a string value
    pub fn scalar_string(value: String, confidence: f32) -> Self {
        Self {
            shape: vec![1],
            data: TensorData::String(vec![value]),
            confidence,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<u32>, confidence: f32) -> Self {
        let size: usize = shape.iter().map(|&d| d as usize).product();
        Self {
            shape,
            data: TensorData::Float(vec![0.0; size]),
            confidence,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<u32>, confidence: f32) -> Self {
        let size: usize = shape.iter().map(|&d| d as usize).product();
        Self {
            shape,
            data: TensorData::Float(vec![1.0; size]),
            confidence,
        }
    }

    /// Create a float tensor from a vector (1D)
    pub fn from_vec(data: Vec<f32>, confidence: f32) -> Self {
        let len = data.len() as u32;
        Self {
            shape: vec![len],
            data: TensorData::Float(data),
            confidence,
        }
    }

    /// Create a string tensor from a vector (1D)
    pub fn from_strings(data: Vec<String>, confidence: f32) -> Self {
        let len = data.len() as u32;
        Self {
            shape: vec![len],
            data: TensorData::String(data),
            confidence,
        }
    }

    /// Create a decimal tensor from a vector (1D)
    pub fn from_decimals(data: Vec<Decimal>, confidence: f32) -> Self {
        let len = data.len() as u32;
        Self {
            shape: vec![len],
            data: TensorData::Decimal(data),
            confidence,
        }
    }

    /// Create a stream tensor from a stream handle
    pub fn from_stream(handle: StreamHandle, confidence: f32) -> Self {
        Self {
            shape: vec![0], // Streams have dynamic shape
            data: TensorData::Stream(handle),
            confidence,
        }
    }

    /// Create a confidence-only scalar (useful for permission checks)
    pub fn confidence_scalar(confidence: f32) -> Self {
        Self {
            shape: vec![1],
            data: TensorData::Float(vec![confidence]),
            confidence,
        }
    }

    /// Get the confidence value of this tensor
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Check if this is a scalar (shape [1])
    pub fn is_scalar(&self) -> bool {
        self.shape == vec![1]
    }

    /// Get scalar value (panics if not a scalar or not float)
    pub fn as_scalar(&self) -> f32 {
        assert!(self.is_scalar(), "Tensor is not a scalar");
        match &self.data {
            TensorData::Float(v) => v[0],
            TensorData::Decimal(v) => {
                use rust_decimal::prelude::ToPrimitive;
                v[0].to_f32().unwrap_or(0.0)
            }
            TensorData::String(_) => panic!("Cannot get scalar from string tensor"),
            TensorData::Stream(_) => panic!("Cannot get scalar from stream tensor"),
        }
    }

    /// Get scalar string value (panics if not a scalar or not string)
    pub fn as_scalar_string(&self) -> &str {
        assert!(self.is_scalar(), "Tensor is not a scalar");
        match &self.data {
            TensorData::String(v) => &v[0],
            _ => panic!("Tensor is not a string tensor"),
        }
    }

    /// Get scalar decimal value (panics if not a scalar or not decimal)
    pub fn as_scalar_decimal(&self) -> Decimal {
        assert!(self.is_scalar(), "Tensor is not a scalar");
        match &self.data {
            TensorData::Decimal(v) => v[0],
            TensorData::Float(v) => Decimal::from_f32_retain(v[0]).unwrap_or(Decimal::ZERO),
            TensorData::String(_) => panic!("Cannot get decimal from string tensor"),
            TensorData::Stream(_) => panic!("Cannot get decimal from stream tensor"),
        }
    }

    /// Compute confidence propagation for binary operations
    /// Using min() to be conservative - result is only as confident as the least confident input
    fn propagate_confidence(a: f32, b: f32) -> f32 {
        a.min(b)
    }

    /// Element-wise operation helper for float tensors
    fn elementwise_op<F>(&self, other: &Tensor, op: F) -> Result<Tensor, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        // Both tensors must be float for element-wise operations
        let self_data = self.data.as_float().ok_or_else(|| TensorError::InvalidShape {
            reason: "Element-wise operations require float tensors".to_string(),
        })?;
        let other_data = other.data.as_float().ok_or_else(|| TensorError::InvalidShape {
            reason: "Element-wise operations require float tensors".to_string(),
        })?;

        // Check shape compatibility
        if self.shape != other.shape {
            // Allow broadcasting for scalars
            if self.is_scalar() {
                let scalar = self_data[0];
                let data: Vec<f32> = other_data.iter().map(|&x| op(scalar, x)).collect();
                return Ok(Tensor {
                    shape: other.shape.clone(),
                    data: TensorData::Float(data),
                    confidence: Self::propagate_confidence(self.confidence, other.confidence),
                });
            } else if other.is_scalar() {
                let scalar = other_data[0];
                let data: Vec<f32> = self_data.iter().map(|&x| op(x, scalar)).collect();
                return Ok(Tensor {
                    shape: self.shape.clone(),
                    data: TensorData::Float(data),
                    confidence: Self::propagate_confidence(self.confidence, other.confidence),
                });
            }
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        let data: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Ok(Tensor {
            shape: self.shape.clone(),
            data: TensorData::Float(data),
            confidence: Self::propagate_confidence(self.confidence, other.confidence),
        })
    }

    /// Checked addition
    pub fn checked_add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| a + b)
    }

    /// Checked subtraction
    pub fn checked_sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| a - b)
    }

    /// Checked multiplication
    pub fn checked_mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| a * b)
    }

    /// Checked division
    pub fn checked_div(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| if b != 0.0 { a / b } else { f32::NAN })
    }

    /// Sum all elements, returning a scalar
    pub fn sum(&self) -> Tensor {
        let float_data = self.float_data();
        let total: f32 = float_data.iter().sum();
        Tensor::scalar(total, self.confidence)
    }

    /// Mean of all elements, returning a scalar
    pub fn mean(&self) -> Tensor {
        let float_data = self.float_data();
        let total: f32 = float_data.iter().sum();
        let count = float_data.len() as f32;
        Tensor::scalar(total / count, self.confidence)
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        let float_data = self.float_data();
        let data: Vec<f32> = float_data.iter().map(|&x| x.max(0.0)).collect();
        Tensor {
            shape: self.shape.clone(),
            data: TensorData::Float(data),
            confidence: self.confidence,
        }
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor {
        let float_data = self.float_data();
        let data: Vec<f32> = float_data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data: TensorData::Float(data),
            confidence: self.confidence,
        }
    }

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        let float_data = self.float_data();
        let data: Vec<f32> = float_data.iter().map(|&x| x.tanh()).collect();
        Tensor {
            shape: self.shape.clone(),
            data: TensorData::Float(data),
            confidence: self.confidence,
        }
    }

    /// Softmax activation (across all elements)
    pub fn softmax(&self) -> Tensor {
        let float_data = self.float_data();
        // Find max for numerical stability
        let max_val = float_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_data: Vec<f32> = float_data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_data.iter().sum();
        let data: Vec<f32> = exp_data.iter().map(|&x| x / sum).collect();
        Tensor {
            shape: self.shape.clone(),
            data: TensorData::Float(data),
            confidence: self.confidence,
        }
    }

    /// Argmax - returns index of maximum value as a scalar
    pub fn argmax(&self) -> Tensor {
        let float_data = self.float_data();
        let (max_idx, _) =
            float_data
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |(max_i, max_v), (i, &v)| {
                    if v > max_v {
                        (i, v)
                    } else {
                        (max_i, max_v)
                    }
                });
        Tensor::scalar(max_idx as f32, self.confidence)
    }

    /// Element-wise equality comparison - returns 1.0 if equal, 0.0 otherwise
    pub fn eq(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| {
            if (a - b).abs() < f32::EPSILON {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Element-wise greater than comparison
    pub fn gt(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    /// Element-wise less than comparison
    pub fn lt(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    /// Matrix multiplication for 2D tensors
    /// Shapes: [M, K] @ [K, N] -> [M, N]
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        // Validate shapes
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "matmul requires 2D tensors, got shapes {:?} and {:?}",
                    self.shape, other.shape
                ),
            });
        }

        let self_data = self.float_data();
        let other_data = other.float_data();

        let m = self.shape[0] as usize;
        let k1 = self.shape[1] as usize;
        let k2 = other.shape[0] as usize;
        let n = other.shape[1] as usize;

        if k1 != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape[0], self.shape[1]],
                got: vec![other.shape[0], other.shape[1]],
            });
        }

        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..k1 {
                    sum += self_data[i * k1 + k] * other_data[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Tensor {
            shape: vec![m as u32, n as u32],
            data: TensorData::Float(result),
            confidence: Self::propagate_confidence(self.confidence, other.confidence),
        })
    }

    /// Reshape tensor to new shape (total elements must match)
    pub fn reshape(&self, new_shape: Vec<u32>) -> Result<Tensor, TensorError> {
        let new_numel: usize = new_shape.iter().map(|&d| d as usize).product();
        if new_numel != self.numel() {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                    self.numel(),
                    new_shape,
                    new_numel
                ),
            });
        }
        Ok(Tensor {
            shape: new_shape,
            data: self.data.clone(),
            confidence: self.confidence,
        })
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                reason: format!("transpose requires 2D tensor, got shape {:?}", self.shape),
            });
        }

        let float_data = self.float_data();
        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;
        let mut result = vec![0.0f32; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = float_data[i * cols + j];
            }
        }

        Ok(Tensor {
            shape: vec![cols as u32, rows as u32],
            data: TensorData::Float(result),
            confidence: self.confidence,
        })
    }

    /// Concatenate tensors along the first axis (float tensors only)
    pub fn concat(tensors: &[&Tensor]) -> Result<Tensor, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidShape {
                reason: "Cannot concatenate empty list of tensors".to_string(),
            });
        }

        // All tensors must have same shape except first dimension
        let first_shape = &tensors[0].shape;
        if first_shape.is_empty() {
            return Err(TensorError::InvalidShape {
                reason: "Cannot concatenate scalar tensors".to_string(),
            });
        }

        let suffix_shape = &first_shape[1..];
        for t in tensors.iter().skip(1) {
            if t.shape.len() != first_shape.len() || &t.shape[1..] != suffix_shape {
                return Err(TensorError::ShapeMismatch {
                    expected: first_shape.clone(),
                    got: t.shape.clone(),
                });
            }
        }

        // Concatenate data
        let mut data = Vec::new();
        let mut min_confidence = 1.0f32;
        let mut first_dim = 0u32;

        for t in tensors {
            data.extend(t.float_data());
            first_dim += t.shape[0];
            min_confidence = min_confidence.min(t.confidence);
        }

        let mut new_shape = vec![first_dim];
        new_shape.extend_from_slice(suffix_shape);

        Ok(Tensor {
            shape: new_shape,
            data: TensorData::Float(data),
            confidence: min_confidence,
        })
    }

    /// Serialize tensor data to bytes (for hashing)
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.data {
            TensorData::Float(v) => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            TensorData::String(v) => v.iter().flat_map(|s| s.as_bytes().to_vec()).collect(),
            TensorData::Decimal(v) => v.iter().flat_map(|d| d.serialize().to_vec()).collect(),
            TensorData::Stream(h) => {
                // Serialize stream handle id and source type
                let mut bytes = h.id.to_le_bytes().to_vec();
                match &h.source_type {
                    StreamSource::WebSocket { url } => {
                        bytes.push(0);
                        bytes.extend(url.as_bytes());
                    }
                    StreamSource::Channel { channel_id } => {
                        bytes.push(1);
                        bytes.extend(channel_id.as_bytes());
                    }
                    StreamSource::Event { event_type } => {
                        bytes.push(2);
                        bytes.extend(event_type.as_bytes());
                    }
                }
                bytes
            }
        }
    }
}

/// Implement Add trait for convenient syntax: tensor_a + tensor_b
impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        Tensor::checked_add(&self, &other).expect("Shape mismatch in Add")
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        Tensor::checked_add(self, other).expect("Shape mismatch in Add")
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        Tensor::checked_sub(&self, &other).expect("Shape mismatch in Sub")
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        Tensor::checked_mul(&self, &other).expect("Shape mismatch in Mul")
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        Tensor::checked_div(&self, &other).expect("Shape mismatch in Div")
    }
}

/// Tensor operation errors
#[derive(Debug, Clone)]
pub enum TensorError {
    ShapeMismatch { expected: Vec<u32>, got: Vec<u32> },
    InvalidShape { reason: String },
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TensorError::InvalidShape { reason } => {
                write!(f, "Invalid shape: {}", reason)
            }
        }
    }
}

impl std::error::Error for TensorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_creation() {
        let t = Tensor::scalar(3.14, 1.0);
        assert_eq!(t.shape, vec![1]);
        assert_eq!(*t.float_data(), vec![3.14]);
        assert!(t.is_scalar());
    }

    #[test]
    fn test_addition() {
        let a = Tensor::scalar(1.0, 1.0);
        let b = Tensor::scalar(2.0, 0.9);
        let c = a + b;
        assert_eq!(c.as_scalar(), 3.0);
        assert_eq!(c.confidence, 0.9); // min(1.0, 0.9)
    }

    #[test]
    fn test_confidence_propagation() {
        let a = Tensor::scalar(5.0, 0.8);
        let b = Tensor::scalar(3.0, 0.6);
        let c = a + b;
        assert_eq!(c.confidence, 0.6); // min(0.8, 0.6)
    }

    #[test]
    fn test_vector_addition() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1.0);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], 1.0);
        let c = a + b;
        assert_eq!(*c.float_data(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scalar_broadcast() {
        let scalar = Tensor::scalar(2.0, 1.0);
        let vec = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1.0);
        let result = scalar.checked_mul(&vec).unwrap();
        assert_eq!(*result.float_data(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_relu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], 1.0);
        let r = t.relu();
        assert_eq!(*r.float_data(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sum() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 0.95);
        let s = t.sum();
        assert_eq!(s.as_scalar(), 10.0);
        assert_eq!(s.confidence, 0.95);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1.0);
        let s = t.softmax();
        // Softmax values should sum to 1
        let sum: f32 = s.float_data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Higher input = higher output
        assert!(s.float_data()[2] > s.float_data()[1]);
        assert!(s.float_data()[1] > s.float_data()[0]);
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0], 0.9);
        let idx = t.argmax();
        assert_eq!(idx.as_scalar(), 1.0); // Index of 5.0
        assert_eq!(idx.confidence, 0.9);
    }

    #[test]
    fn test_matmul() {
        // [2, 3] @ [3, 2] = [2, 2]
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0);
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.9);
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape, vec![2, 2]);
        // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert_eq!(*c.float_data(), vec![22.0, 28.0, 49.0, 64.0]);
        assert_eq!(c.confidence, 0.9);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0);
        let r = t.reshape(vec![2, 3]).unwrap();
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(r.data, t.data);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0);
        let r = t.transpose().unwrap();
        assert_eq!(r.shape, vec![3, 2]);
        // Original: [[1,2,3], [4,5,6]]
        // Transposed: [[1,4], [2,5], [3,6]]
        assert_eq!(*r.float_data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_concat() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0);
        let b = Tensor::new(vec![1, 3], vec![7.0, 8.0, 9.0], 0.8);
        let c = Tensor::concat(&[&a, &b]).unwrap();

        assert_eq!(c.shape, vec![3, 3]);
        assert_eq!(*c.float_data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(c.confidence, 0.8); // min(1.0, 0.8)
    }

    #[test]
    fn test_comparisons() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1.0);
        let b = Tensor::from_vec(vec![2.0, 2.0, 1.0], 1.0);

        let eq = a.eq(&b).unwrap();
        assert_eq!(*eq.float_data(), vec![0.0, 1.0, 0.0]);

        let gt = a.gt(&b).unwrap();
        assert_eq!(*gt.float_data(), vec![0.0, 0.0, 1.0]);

        let lt = a.lt(&b).unwrap();
        assert_eq!(*lt.float_data(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_string_tensor() {
        let t = Tensor::from_strings(vec!["BTC/USD".to_string(), "ETH/USD".to_string()], 1.0);
        assert_eq!(t.shape, vec![2]);
        assert!(t.data.is_string());
        let strings = t.data.as_string().unwrap();
        assert_eq!(strings[0], "BTC/USD");
        assert_eq!(strings[1], "ETH/USD");
    }

    #[test]
    fn test_decimal_tensor() {
        let t = Tensor::from_decimals(vec![Decimal::new(12345, 2), Decimal::new(67890, 3)], 0.95);
        assert_eq!(t.shape, vec![2]);
        assert!(t.data.is_decimal());
        let decimals = t.data.as_decimal().unwrap();
        assert_eq!(decimals[0], Decimal::new(12345, 2)); // 123.45
        assert_eq!(decimals[1], Decimal::new(67890, 3)); // 67.890
    }

    #[test]
    fn test_stream_tensor() {
        let handle = StreamHandle {
            id: 42,
            source_type: StreamSource::WebSocket { url: "wss://example.com".to_string() },
            buffer: Arc::new(RwLock::new(VecDeque::new())),
        };
        let t = Tensor::from_stream(handle.clone(), 1.0);
        assert!(t.data.is_stream());
        let stream = t.data.as_stream().unwrap();
        assert_eq!(stream.id, 42);
        assert!(matches!(stream.source_type, StreamSource::WebSocket { .. }));
    }

    #[test]
    fn test_confidence_scalar() {
        let t = Tensor::confidence_scalar(0.85);
        assert!(t.is_scalar());
        assert_eq!(t.as_scalar(), 0.85);
        assert_eq!(t.confidence(), 0.85);
    }
}
