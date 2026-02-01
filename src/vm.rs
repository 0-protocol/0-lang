//! VM - The ZeroLang Virtual Machine
//!
//! Executes Zero graphs by topologically sorting nodes and
//! evaluating operations on tensors.

use std::collections::HashMap;
use std::sync::Arc;

use crate::graph::{GraphError, NodeHash, Op, RuntimeGraph, RuntimeNode};
use crate::Tensor;

/// Trait for resolving external node calls.
///
/// Implement this trait to provide custom behavior for external URIs.
/// The resolver receives the URI and input tensors, and returns output tensor(s).
pub trait ExternalResolver: Send + Sync {
    /// Resolve an external call.
    ///
    /// # Arguments
    /// * `uri` - The URI identifying the external resource (e.g., "ffi:rust:my_func")
    /// * `inputs` - Input tensors from the graph
    ///
    /// # Returns
    /// A tensor result or an error message
    fn resolve(&self, uri: &str, inputs: Vec<&Tensor>) -> Result<Tensor, String>;
}

/// A resolver that rejects all external calls (default safe behavior)
pub struct RejectingResolver;

impl ExternalResolver for RejectingResolver {
    fn resolve(&self, uri: &str, _inputs: Vec<&Tensor>) -> Result<Tensor, String> {
        Err(format!(
            "External node with URI '{}' cannot be resolved. \
            Configure an ExternalResolver or use --unsafe to skip external nodes.",
            uri
        ))
    }
}

/// A resolver that returns a zero tensor for any external call (for testing)
pub struct MockResolver {
    /// The shape of tensors to return
    pub output_shape: Vec<u32>,
    /// The confidence to use
    pub confidence: f32,
}

impl Default for MockResolver {
    fn default() -> Self {
        Self {
            output_shape: vec![1],
            confidence: 1.0,
        }
    }
}

impl ExternalResolver for MockResolver {
    fn resolve(&self, _uri: &str, _inputs: Vec<&Tensor>) -> Result<Tensor, String> {
        Ok(Tensor::zeros(self.output_shape.clone(), self.confidence))
    }
}

/// The ZeroLang Virtual Machine
pub struct VM {
    /// Memory: stores computed tensor values by node hash
    memory: HashMap<NodeHash, Tensor>,
    /// Execution fuel (max operations before halt)
    fuel: u64,
    /// Operations executed
    ops_executed: u64,
    /// External resolver for handling External nodes
    external_resolver: Option<Arc<dyn ExternalResolver>>,
}

impl VM {
    /// Create a new VM with default fuel
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            fuel: 1_000_000, // Default: 1 million operations
            ops_executed: 0,
            external_resolver: None,
        }
    }

    /// Create a VM with specified fuel budget
    pub fn with_fuel(fuel: u64) -> Self {
        Self {
            memory: HashMap::new(),
            fuel,
            ops_executed: 0,
            external_resolver: None,
        }
    }

    /// Create a VM with fuel from graph's halting proof (or default if none)
    pub fn from_graph(graph: &RuntimeGraph) -> Self {
        if let Some((_max_steps, fuel_budget)) = graph.get_halting_proof() {
            Self::with_fuel(fuel_budget)
        } else {
            Self::new()
        }
    }

    /// Set an external resolver for handling External nodes
    pub fn with_external_resolver(mut self, resolver: Arc<dyn ExternalResolver>) -> Self {
        self.external_resolver = Some(resolver);
        self
    }

    /// Check if this graph contains external nodes
    pub fn graph_has_external_nodes(graph: &RuntimeGraph) -> bool {
        graph
            .nodes
            .values()
            .any(|node| matches!(node, RuntimeNode::External { .. }))
    }

    /// Execute a graph and return the output tensors
    pub fn execute(&mut self, graph: &RuntimeGraph) -> Result<Vec<Tensor>, VMError> {
        // Clear memory from previous executions
        self.memory.clear();
        self.ops_executed = 0;

        // Get topological order
        let order = graph.topological_sort().map_err(VMError::GraphError)?;

        // Execute nodes in order
        for hash in order {
            self.execute_node(&hash, graph)?;
        }

        // Collect outputs
        let mut outputs = Vec::new();
        for output_hash in &graph.outputs {
            let tensor = self
                .memory
                .get(output_hash)
                .ok_or_else(|| VMError::NodeNotComputed(hex::encode(output_hash)))?
                .clone();
            outputs.push(tensor);
        }

        Ok(outputs)
    }

    /// Execute a single node
    fn execute_node(&mut self, hash: &NodeHash, graph: &RuntimeGraph) -> Result<(), VMError> {
        // Check fuel
        if self.ops_executed >= self.fuel {
            return Err(VMError::OutOfFuel);
        }

        let node = graph
            .nodes
            .get(hash)
            .ok_or_else(|| VMError::NodeNotFound(hex::encode(hash)))?;

        let result = match node {
            RuntimeNode::Constant(tensor) => {
                // Constants just get stored directly
                tensor.clone()
            }
            RuntimeNode::Operation { op, inputs } => {
                self.ops_executed += 1;
                self.execute_operation(*op, inputs)?
            }
            RuntimeNode::Branch {
                condition,
                threshold,
                true_branch,
                false_branch,
            } => {
                self.ops_executed += 1;
                self.execute_branch(condition, *threshold, true_branch, false_branch)?
            }
            RuntimeNode::External { uri, inputs } => {
                self.ops_executed += 1;
                self.execute_external(uri, inputs)?
            }
        };

        self.memory.insert(hash.clone(), result);
        Ok(())
    }

    /// Execute an operation on input tensors
    fn execute_operation(&self, op: Op, inputs: &[NodeHash]) -> Result<Tensor, VMError> {
        // Fetch input tensors
        let input_tensors: Result<Vec<&Tensor>, VMError> = inputs
            .iter()
            .map(|h| {
                self.memory
                    .get(h)
                    .ok_or_else(|| VMError::NodeNotComputed(hex::encode(h)))
            })
            .collect();
        let input_tensors = input_tensors?;

        match op {
            // Binary operations
            Op::Add => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .checked_add(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Sub => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .checked_sub(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Mul => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .checked_mul(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Div => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .checked_div(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }

            // Unary operations
            Op::Relu => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].relu())
            }
            Op::Sigmoid => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].sigmoid())
            }
            Op::Tanh => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].tanh())
            }

            // Reductions
            Op::Sum => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].sum())
            }
            Op::Mean => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].mean())
            }

            // Special
            Op::Identity => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].clone())
            }

            // Matrix multiplication
            Op::Matmul => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .matmul(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }

            // Softmax activation
            Op::Softmax => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].softmax())
            }

            // Comparison operations
            Op::Eq => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .eq(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Gt => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .gt(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Lt => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .lt(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Gte => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                // Gte = NOT Lt
                let lt_result = input_tensors[0]
                    .lt(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))?;
                // Invert: 1.0 - result (since comparison returns 0 or 1)
                let inverted: Vec<f32> = lt_result.data.iter().map(|&v| 1.0 - v).collect();
                Ok(Tensor::new(lt_result.shape, inverted, lt_result.confidence))
            }
            Op::Lte => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                // Lte = NOT Gt
                let gt_result = input_tensors[0]
                    .gt(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))?;
                let inverted: Vec<f32> = gt_result.data.iter().map(|&v| 1.0 - v).collect();
                Ok(Tensor::new(gt_result.shape, inverted, gt_result.confidence))
            }

            // Argmax
            Op::Argmax => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].argmax())
            }

            // Min - element-wise minimum of two tensors, or reduction if single input
            Op::Min => {
                if input_tensors.len() == 1 {
                    // Reduction: find minimum value
                    let min_val = input_tensors[0]
                        .data
                        .iter()
                        .cloned()
                        .fold(f32::INFINITY, f32::min);
                    Ok(Tensor::scalar(min_val, input_tensors[0].confidence))
                } else if input_tensors.len() == 2 {
                    // Element-wise minimum
                    if input_tensors[0].shape != input_tensors[1].shape {
                        return Err(VMError::TensorError("Shape mismatch for Min".to_string()));
                    }
                    let data: Vec<f32> = input_tensors[0]
                        .data
                        .iter()
                        .zip(input_tensors[1].data.iter())
                        .map(|(&a, &b)| a.min(b))
                        .collect();
                    let confidence = input_tensors[0].confidence.min(input_tensors[1].confidence);
                    Ok(Tensor::new(input_tensors[0].shape.clone(), data, confidence))
                } else {
                    Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    })
                }
            }

            // Max - element-wise maximum of two tensors, or reduction if single input
            Op::Max => {
                if input_tensors.len() == 1 {
                    // Reduction: find maximum value
                    let max_val = input_tensors[0]
                        .data
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    Ok(Tensor::scalar(max_val, input_tensors[0].confidence))
                } else if input_tensors.len() == 2 {
                    // Element-wise maximum
                    if input_tensors[0].shape != input_tensors[1].shape {
                        return Err(VMError::TensorError("Shape mismatch for Max".to_string()));
                    }
                    let data: Vec<f32> = input_tensors[0]
                        .data
                        .iter()
                        .zip(input_tensors[1].data.iter())
                        .map(|(&a, &b)| a.max(b))
                        .collect();
                    let confidence = input_tensors[0].confidence.min(input_tensors[1].confidence);
                    Ok(Tensor::new(input_tensors[0].shape.clone(), data, confidence))
                } else {
                    Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    })
                }
            }

            // Shape manipulation
            Op::Reshape => {
                // Reshape needs the target shape as the second input
                // For now, we'll treat the second tensor's data as shape
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                let new_shape: Vec<u32> = input_tensors[1].data.iter().map(|&x| x as u32).collect();
                input_tensors[0]
                    .reshape(new_shape)
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Transpose => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .transpose()
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Concat => {
                if input_tensors.is_empty() {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: 0,
                    });
                }
                Tensor::concat(&input_tensors).map_err(|e| VMError::TensorError(e.to_string()))
            }

            // Embed - converts a hash to a deterministic embedding
            // For now, we create a simple deterministic embedding from the input
            Op::Embed => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                // Create a 768-dim embedding from the input tensor's data
                // This is a placeholder - a real implementation might use a lookup table
                let input_len = input_tensors[0].data.len().min(768);
                let mut embedding = vec![0.0f32; 768];
                embedding[..input_len].copy_from_slice(&input_tensors[0].data[..input_len]);
                // Fill remaining with deterministic pattern based on existing values
                for (i, val) in embedding.iter_mut().enumerate().skip(input_len) {
                    *val = ((i as f32) * 0.1).sin() * 0.5 + 0.5;
                }
                Ok(Tensor {
                    shape: vec![768],
                    data: embedding,
                    confidence: input_tensors[0].confidence,
                })
            }

            // Abs - absolute value
            Op::Abs => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                let data: Vec<f32> = input_tensors[0].data.iter().map(|&v| v.abs()).collect();
                Ok(Tensor::new(
                    input_tensors[0].shape.clone(),
                    data,
                    input_tensors[0].confidence,
                ))
            }

            // Neg - negation
            Op::Neg => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                let data: Vec<f32> = input_tensors[0].data.iter().map(|&v| -v).collect();
                Ok(Tensor::new(
                    input_tensors[0].shape.clone(),
                    data,
                    input_tensors[0].confidence,
                ))
            }

            // Clamp - clamp values to [min, max] range
            // Inputs: [tensor, min_tensor, max_tensor]
            Op::Clamp => {
                if input_tensors.len() != 3 {
                    return Err(VMError::WrongInputCount {
                        expected: 3,
                        got: input_tensors.len(),
                    });
                }
                let min_val = input_tensors[1].as_scalar();
                let max_val = input_tensors[2].as_scalar();
                let data: Vec<f32> = input_tensors[0]
                    .data
                    .iter()
                    .map(|&v| v.clamp(min_val, max_val))
                    .collect();
                Ok(Tensor::new(
                    input_tensors[0].shape.clone(),
                    data,
                    input_tensors[0].confidence,
                ))
            }
        }
    }

    /// Execute an external node
    fn execute_external(&self, uri: &str, inputs: &[NodeHash]) -> Result<Tensor, VMError> {
        // Fetch input tensors
        let input_tensors: Result<Vec<&Tensor>, VMError> = inputs
            .iter()
            .map(|h| {
                self.memory
                    .get(h)
                    .ok_or_else(|| VMError::NodeNotComputed(hex::encode(h)))
            })
            .collect();
        let input_tensors = input_tensors?;

        // Use the resolver if available
        match &self.external_resolver {
            Some(resolver) => resolver.resolve(uri, input_tensors).map_err(|e| {
                VMError::ExternalResolutionFailed {
                    uri: uri.to_string(),
                    reason: e,
                }
            }),
            None => Err(VMError::ExternalNodeRequiresResolver(uri.to_string())),
        }
    }

    /// Execute a branch node
    fn execute_branch(
        &self,
        condition: &NodeHash,
        threshold: f32,
        true_branch: &NodeHash,
        false_branch: &NodeHash,
    ) -> Result<Tensor, VMError> {
        // Get condition tensor (must be scalar)
        let cond_tensor = self
            .memory
            .get(condition)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(condition)))?;

        if !cond_tensor.is_scalar() {
            return Err(VMError::BranchConditionNotScalar);
        }

        let cond_value = cond_tensor.as_scalar();

        // Choose branch based on threshold
        let branch_hash = if cond_value >= threshold {
            true_branch
        } else {
            false_branch
        };

        // Return the tensor from the chosen branch
        self.memory
            .get(branch_hash)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(branch_hash)))
            .cloned()
    }

    /// Get the number of operations executed
    pub fn ops_executed(&self) -> u64 {
        self.ops_executed
    }

    /// Get remaining fuel
    pub fn remaining_fuel(&self) -> u64 {
        self.fuel.saturating_sub(self.ops_executed)
    }
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

/// VM execution errors
#[derive(Debug)]
pub enum VMError {
    GraphError(GraphError),
    NodeNotFound(String),
    NodeNotComputed(String),
    OutOfFuel,
    WrongInputCount {
        expected: usize,
        got: usize,
    },
    TensorError(String),
    BranchConditionNotScalar,
    /// Operation not yet implemented
    UnimplementedOperation(String),
    /// External node requires a resolver
    ExternalNodeRequiresResolver(String),
    /// External node resolution failed
    ExternalResolutionFailed {
        uri: String,
        reason: String,
    },
}

impl std::fmt::Display for VMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VMError::GraphError(e) => write!(f, "Graph error: {}", e),
            VMError::NodeNotFound(h) => write!(f, "Node not found: {}", h),
            VMError::NodeNotComputed(h) => write!(f, "Node not computed: {}", h),
            VMError::OutOfFuel => write!(f, "Out of fuel (exceeded max operations)"),
            VMError::WrongInputCount { expected, got } => {
                write!(f, "Wrong input count: expected {}, got {}", expected, got)
            }
            VMError::TensorError(e) => write!(f, "Tensor error: {}", e),
            VMError::BranchConditionNotScalar => write!(f, "Branch condition must be a scalar"),
            VMError::UnimplementedOperation(op) => {
                write!(f, "Operation not yet implemented: {}", op)
            }
            VMError::ExternalNodeRequiresResolver(uri) => {
                write!(f, "External node requires resolver: {}", uri)
            }
            VMError::ExternalResolutionFailed { uri, reason } => {
                write!(f, "External resolution failed for '{}': {}", uri, reason)
            }
        }
    }
}

impl std::error::Error for VMError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RuntimeNode;

    #[test]
    fn test_external_node_with_mock_resolver() {
        let uri = "test:mock".to_string();
        let input_tensor = Tensor::scalar(1.0, 1.0);
        let input_hash = vec![1, 2, 3];
        let external_hash = vec![4, 5, 6];

        let mut nodes = HashMap::new();
        nodes.insert(input_hash.clone(), RuntimeNode::Constant(input_tensor));
        nodes.insert(
            external_hash.clone(),
            RuntimeNode::External {
                uri,
                inputs: vec![input_hash.clone()],
            },
        );

        let graph = RuntimeGraph {
            nodes,
            entry_point: input_hash,
            outputs: vec![external_hash],
            version: 0,
            proofs: vec![],
        };

        // Without resolver, should fail
        let mut vm = VM::new();
        let result = vm.execute(&graph);
        assert!(matches!(
            result,
            Err(VMError::ExternalNodeRequiresResolver(_))
        ));

        // With mock resolver, should succeed
        let resolver = Arc::new(MockResolver::default());
        let mut vm = VM::new().with_external_resolver(resolver);
        let result = vm.execute(&graph);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_vm_constant_execution() {
        let mut graph = RuntimeGraph {
            nodes: HashMap::new(),
            entry_point: vec![1, 2, 3],
            outputs: vec![vec![1, 2, 3]],
            version: 0,
            proofs: vec![],
        };

        graph.nodes.insert(
            vec![1, 2, 3],
            RuntimeNode::Constant(Tensor::scalar(42.0, 1.0)),
        );

        let mut vm = VM::new();
        let outputs = vm.execute(&graph).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_scalar(), 42.0);
    }

    #[test]
    fn test_vm_addition() {
        let mut graph = RuntimeGraph {
            nodes: HashMap::new(),
            entry_point: vec![1],
            outputs: vec![vec![3]],
            version: 0,
            proofs: vec![],
        };

        // Node 1: Constant 10.0
        graph
            .nodes
            .insert(vec![1], RuntimeNode::Constant(Tensor::scalar(10.0, 1.0)));

        // Node 2: Constant 20.0
        graph
            .nodes
            .insert(vec![2], RuntimeNode::Constant(Tensor::scalar(20.0, 0.9)));

        // Node 3: Add Node1 + Node2
        graph.nodes.insert(
            vec![3],
            RuntimeNode::Operation {
                op: Op::Add,
                inputs: vec![vec![1], vec![2]],
            },
        );

        let mut vm = VM::new();
        let outputs = vm.execute(&graph).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_scalar(), 30.0);
        assert_eq!(outputs[0].confidence, 0.9); // min(1.0, 0.9)
    }
}
