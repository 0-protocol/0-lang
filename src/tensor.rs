//! Tensor - The fundamental data type in ZeroLang
//!
//! Replaces int, float, bool, string with probabilistic vectors.
//! Every value carries a confidence score.

use std::ops::{Add, Div, Mul, Sub};

/// The fundamental data type in ZeroLang.
///
/// In traditional languages, you have `int`, `float`, `bool`, `string`.
/// In ZeroLang, everything is a Tensor with a confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Dimensions of the tensor, e.g., [768] for embedding, [1] for scalar
    pub shape: Vec<u32>,
    /// Flattened tensor data
    pub data: Vec<f32>,
    /// Meta-confidence in this tensor's validity [0.0, 1.0]
    /// This is the "Schrödinger" probability - how certain are we about this value?
    pub confidence: f32,
}

impl Tensor {
    /// Create a new tensor with given shape, data, and confidence
    pub fn new(shape: Vec<u32>, data: Vec<f32>, confidence: f32) -> Self {
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
            data: vec![value],
            confidence,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<u32>, confidence: f32) -> Self {
        let size: usize = shape.iter().map(|&d| d as usize).product();
        Self {
            shape,
            data: vec![0.0; size],
            confidence,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<u32>, confidence: f32) -> Self {
        let size: usize = shape.iter().map(|&d| d as usize).product();
        Self {
            shape,
            data: vec![1.0; size],
            confidence,
        }
    }

    /// Create a tensor from a vector (1D)
    pub fn from_vec(data: Vec<f32>, confidence: f32) -> Self {
        let len = data.len() as u32;
        Self {
            shape: vec![len],
            data,
            confidence,
        }
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Check if this is a scalar (shape [1])
    pub fn is_scalar(&self) -> bool {
        self.shape == vec![1]
    }

    /// Get scalar value (panics if not a scalar)
    pub fn as_scalar(&self) -> f32 {
        assert!(self.is_scalar(), "Tensor is not a scalar");
        self.data[0]
    }

    /// Compute confidence propagation for binary operations
    /// Using min() to be conservative - result is only as confident as the least confident input
    fn propagate_confidence(a: f32, b: f32) -> f32 {
        a.min(b)
    }

    /// Element-wise operation helper
    fn elementwise_op<F>(&self, other: &Tensor, op: F) -> Result<Tensor, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        // Check shape compatibility
        if self.shape != other.shape {
            // Allow broadcasting for scalars
            if self.is_scalar() {
                let scalar = self.data[0];
                let data: Vec<f32> = other.data.iter().map(|&x| op(scalar, x)).collect();
                return Ok(Tensor {
                    shape: other.shape.clone(),
                    data,
                    confidence: Self::propagate_confidence(self.confidence, other.confidence),
                });
            } else if other.is_scalar() {
                let scalar = other.data[0];
                let data: Vec<f32> = self.data.iter().map(|&x| op(x, scalar)).collect();
                return Ok(Tensor {
                    shape: self.shape.clone(),
                    data,
                    confidence: Self::propagate_confidence(self.confidence, other.confidence),
                });
            }
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Ok(Tensor {
            shape: self.shape.clone(),
            data,
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
        let total: f32 = self.data.iter().sum();
        Tensor::scalar(total, self.confidence)
    }

    /// Mean of all elements, returning a scalar
    pub fn mean(&self) -> Tensor {
        let total: f32 = self.data.iter().sum();
        let count = self.data.len() as f32;
        Tensor::scalar(total / count, self.confidence)
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            confidence: self.confidence,
        }
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            confidence: self.confidence,
        }
    }

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            confidence: self.confidence,
        }
    }

    /// Softmax activation (across all elements)
    pub fn softmax(&self) -> Tensor {
        // Find max for numerical stability
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_data: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_data.iter().sum();
        let data: Vec<f32> = exp_data.iter().map(|&x| x / sum).collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            confidence: self.confidence,
        }
    }

    /// Argmax - returns index of maximum value as a scalar
    pub fn argmax(&self) -> Tensor {
        let (max_idx, _) =
            self.data
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
                    sum += self.data[i * k1 + k] * other.data[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Tensor {
            shape: vec![m as u32, n as u32],
            data: result,
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

        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;
        let mut result = vec![0.0f32; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Tensor {
            shape: vec![cols as u32, rows as u32],
            data: result,
            confidence: self.confidence,
        })
    }

    /// Concatenate tensors along the first axis
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
            data.extend(&t.data);
            first_dim += t.shape[0];
            min_confidence = min_confidence.min(t.confidence);
        }

        let mut new_shape = vec![first_dim];
        new_shape.extend_from_slice(suffix_shape);

        Ok(Tensor {
            shape: new_shape,
            data,
            confidence: min_confidence,
        })
    }

    /// Serialize tensor data to bytes (for hashing)
    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.iter().flat_map(|f| f.to_le_bytes()).collect()
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
        assert_eq!(t.data, vec![3.14]);
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
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scalar_broadcast() {
        let scalar = Tensor::scalar(2.0, 1.0);
        let vec = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1.0);
        let result = scalar.checked_mul(&vec).unwrap();
        assert_eq!(result.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_relu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], 1.0);
        let r = t.relu();
        assert_eq!(r.data, vec![0.0, 0.0, 1.0, 2.0]);
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
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Higher input = higher output
        assert!(s.data[2] > s.data[1]);
        assert!(s.data[1] > s.data[0]);
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
        assert_eq!(c.data, vec![22.0, 28.0, 49.0, 64.0]);
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
        assert_eq!(r.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_concat() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0);
        let b = Tensor::new(vec![1, 3], vec![7.0, 8.0, 9.0], 0.8);
        let c = Tensor::concat(&[&a, &b]).unwrap();

        assert_eq!(c.shape, vec![3, 3]);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(c.confidence, 0.8); // min(1.0, 0.8)
    }

    #[test]
    fn test_comparisons() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1.0);
        let b = Tensor::from_vec(vec![2.0, 2.0, 1.0], 1.0);

        let eq = a.eq(&b).unwrap();
        assert_eq!(eq.data, vec![0.0, 1.0, 0.0]);

        let gt = a.gt(&b).unwrap();
        assert_eq!(gt.data, vec![0.0, 0.0, 1.0]);

        let lt = a.lt(&b).unwrap();
        assert_eq!(lt.data, vec![1.0, 0.0, 0.0]);
    }
}
