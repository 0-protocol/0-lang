//! VM - The ZeroLang Virtual Machine
//!
//! Executes Zero graphs by topologically sorting nodes and
//! evaluating operations on tensors.

use std::collections::HashMap;
use std::sync::Arc;

use crate::graph::{GraphError, NodeHash, Op, RuntimeGraph, RuntimeNode};
use crate::permission::{evaluate_permission, PermissionPolicy};
use crate::route::Router;
use crate::stdlib::json::{json_array, json_get, json_parse};
use crate::tensor::TensorData;
use crate::web3::{get_gas_price_json, oracle_read_json};
use crate::Tensor;

/// Execution trace for proof-carrying actions
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Ordered list of node hashes executed
    pub nodes: Vec<[u8; 32]>,
    /// Timestamp of each execution (milliseconds since epoch)
    pub timestamps: Vec<u64>,
    /// Confidence score at each step
    pub confidences: Vec<f32>,
    /// Node types executed (for debugging)
    pub node_types: Vec<String>,
}

impl ExecutionTrace {
    /// Create a new empty execution trace
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            timestamps: Vec::new(),
            confidences: Vec::new(),
            node_types: Vec::new(),
        }
    }

    /// Add an entry to the trace
    pub fn record(&mut self, hash: &NodeHash, confidence: f32, node_type: &str) {
        let mut hash_array = [0u8; 32];
        let len = hash.len().min(32);
        hash_array[..len].copy_from_slice(&hash[..len]);
        
        self.nodes.push(hash_array);
        self.timestamps.push(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );
        self.confidences.push(confidence);
        self.node_types.push(node_type.to_string());
    }

    /// Get the number of nodes executed
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if trace is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get minimum confidence in the trace
    pub fn min_confidence(&self) -> f32 {
        self.confidences.iter().cloned().fold(1.0f32, f32::min)
    }

    /// Get average confidence in the trace
    pub fn avg_confidence(&self) -> f32 {
        if self.confidences.is_empty() {
            return 1.0;
        }
        self.confidences.iter().sum::<f32>() / self.confidences.len() as f32
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Confidence combination strategy for ConfidenceCombine operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfidenceCombineStrategy {
    /// Minimum of all inputs (most conservative)
    Min,
    /// Maximum of all inputs (most optimistic)
    Max,
    /// Product of all inputs
    Product,
    /// Arithmetic mean
    Average,
    /// Geometric mean (default)
    GeometricMean,
}

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
    /// Persistent state storage (key -> tensor)
    state: HashMap<String, Tensor>,
}

impl VM {
    /// Create a new VM with default fuel
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            fuel: 1_000_000, // Default: 1 million operations
            ops_executed: 0,
            external_resolver: None,
            state: HashMap::new(),
        }
    }

    /// Create a VM with specified fuel budget
    pub fn with_fuel(fuel: u64) -> Self {
        Self {
            memory: HashMap::new(),
            fuel,
            ops_executed: 0,
            external_resolver: None,
            state: HashMap::new(),
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

    /// Get a reference to the VM's persistent state
    pub fn get_state(&self) -> &HashMap<String, Tensor> {
        &self.state
    }

    /// Get a mutable reference to the VM's persistent state
    pub fn get_state_mut(&mut self) -> &mut HashMap<String, Tensor> {
        &mut self.state
    }

    /// Set a state value
    pub fn set_state(&mut self, key: String, value: Tensor) {
        self.state.insert(key, value);
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

    /// Execute a graph and return both results and execution trace
    pub fn execute_with_trace(
        &mut self,
        graph: &RuntimeGraph,
    ) -> Result<(Vec<Tensor>, ExecutionTrace), VMError> {
        // Clear memory from previous executions
        self.memory.clear();
        self.ops_executed = 0;
        let mut trace = ExecutionTrace::new();

        // Get topological order
        let order = graph.topological_sort().map_err(VMError::GraphError)?;

        // Execute nodes in order, recording trace
        for hash in order {
            self.execute_node(&hash, graph)?;
            
            // Record trace entry
            if let Some(tensor) = self.memory.get(&hash) {
                let node = graph.nodes.get(&hash);
                let node_type = match node {
                    Some(RuntimeNode::Constant(_)) => "Constant",
                    Some(RuntimeNode::Operation { op, .. }) => match op {
                        Op::Add => "Op::Add",
                        Op::Sub => "Op::Sub",
                        Op::Mul => "Op::Mul",
                        Op::Div => "Op::Div",
                        Op::ConfidenceCombine => "Op::ConfidenceCombine",
                        Op::ConfidenceThreshold => "Op::ConfidenceThreshold",
                        Op::ConfidenceDecay => "Op::ConfidenceDecay",
                        Op::ConfidenceBoost => "Op::ConfidenceBoost",
                        Op::OracleRead => "Op::OracleRead",
                        Op::GetGasPrice => "Op::GetGasPrice",
                        _ => "Operation",
                    },
                    Some(RuntimeNode::Branch { .. }) => "Branch",
                    Some(RuntimeNode::External { .. }) => "External",
                    Some(RuntimeNode::State { .. }) => "State",
                    Some(RuntimeNode::Permission { .. }) => "Permission",
                    Some(RuntimeNode::Route { .. }) => "Route",
                    Some(RuntimeNode::Timer { .. }) => "Timer",
                    None => "Unknown",
                };
                trace.record(&hash, tensor.confidence, node_type);
            }
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

        Ok((outputs, trace))
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
            RuntimeNode::State { key, default } => {
                // State nodes return persisted value or default
                self.state.get(key).cloned().unwrap_or_else(|| default.clone())
            }
            RuntimeNode::Permission {
                subject,
                action,
                threshold,
                fallback,
            } => {
                self.ops_executed += 1;
                self.execute_permission(subject, action, *threshold, fallback)?
            }
            RuntimeNode::Route {
                input,
                routes,
                default,
            } => {
                self.ops_executed += 1;
                self.execute_route(input, routes, default)?
            }
            RuntimeNode::Timer {
                schedule: _,
                target,
                max_concurrent: _,
                overlap_policy: _,
            } => {
                // Timer nodes just return the target node's value when evaluated
                // The actual scheduling is handled by the TimerManager
                self.memory
                    .get(target)
                    .ok_or_else(|| VMError::NodeNotComputed(hex::encode(target)))?
                    .clone()
            }
        };

        self.memory.insert(hash.clone(), result);
        Ok(())
    }

    /// Execute a permission check node
    fn execute_permission(
        &self,
        subject: &NodeHash,
        action: &NodeHash,
        threshold: f32,
        fallback: &NodeHash,
    ) -> Result<Tensor, VMError> {
        let subject_tensor = self
            .memory
            .get(subject)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(subject)))?;
        let action_tensor = self
            .memory
            .get(action)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(action)))?;
        let fallback_tensor = self
            .memory
            .get(fallback)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(fallback)))?;

        // Evaluate permission
        let policy = PermissionPolicy::with_threshold(threshold);
        let result = evaluate_permission(subject_tensor, action_tensor, &policy);

        if result.allowed {
            // Return a confidence tensor indicating permission granted
            Ok(Tensor::confidence_scalar(result.confidence))
        } else {
            // Return the fallback tensor
            Ok(fallback_tensor.clone())
        }
    }

    /// Execute a route node
    fn execute_route(
        &self,
        input: &NodeHash,
        routes: &[crate::graph::Route],
        default: &NodeHash,
    ) -> Result<Tensor, VMError> {
        let _input_tensor = self
            .memory
            .get(input)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(input)))?;

        // Evaluate each route's condition
        let mut condition_results = Vec::new();
        for route in routes {
            let condition_tensor = self
                .memory
                .get(&route.condition)
                .ok_or_else(|| VMError::NodeNotComputed(hex::encode(&route.condition)))?;
            condition_results.push(condition_tensor.confidence);
        }

        // Create router and evaluate
        let router = Router::from_graph_routes(routes.to_vec(), default.clone());
        let result = router.evaluate(&condition_results);

        // Get the target tensor
        let target_hash = if result.used_default {
            default
        } else {
            &result.selected_route.unwrap().target
        };

        self.memory
            .get(target_hash)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(target_hash)))
            .cloned()
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
                let inverted: Vec<f32> = lt_result.float_data().iter().map(|&v| 1.0 - v).collect();
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
                let inverted: Vec<f32> = gt_result.float_data().iter().map(|&v| 1.0 - v).collect();
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
                        .float_data()
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
                        .float_data()
                        .iter()
                        .zip(input_tensors[1].float_data().iter())
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
                        .float_data()
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
                        .float_data()
                        .iter()
                        .zip(input_tensors[1].float_data().iter())
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
                let new_shape: Vec<u32> = input_tensors[1].float_data().iter().map(|&x| x as u32).collect();
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
                let float_data = input_tensors[0].float_data();
                let input_len = float_data.len().min(768);
                let mut embedding = vec![0.0f32; 768];
                embedding[..input_len].copy_from_slice(&float_data[..input_len]);
                // Fill remaining with deterministic pattern based on existing values
                for (i, val) in embedding.iter_mut().enumerate().skip(input_len) {
                    *val = ((i as f32) * 0.1).sin() * 0.5 + 0.5;
                }
                Ok(Tensor::new(
                    vec![768],
                    embedding,
                    input_tensors[0].confidence,
                ))
            }

            // Abs - absolute value
            Op::Abs => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                let data: Vec<f32> = input_tensors[0].float_data().iter().map(|&v| v.abs()).collect();
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
                let data: Vec<f32> = input_tensors[0].float_data().iter().map(|&v| -v).collect();
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
                    .float_data()
                    .iter()
                    .map(|&v| v.clamp(min_val, max_val))
                    .collect();
                Ok(Tensor::new(
                    input_tensors[0].shape.clone(),
                    data,
                    input_tensors[0].confidence,
                ))
            }

            // JSON Operations
            Op::JsonParse => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                json_parse(input_tensors[0])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }

            Op::JsonGet => {
                // JsonGet expects 2 inputs: JSON tensor and key path tensor
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                let key_path = match &input_tensors[1].data {
                    TensorData::String(v) if !v.is_empty() => v[0].clone(),
                    _ => return Err(VMError::TensorError("JsonGet requires string key path".to_string())),
                };
                json_get(input_tensors[0], &key_path)
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }

            Op::JsonArray => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                json_array(input_tensors[0])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }

            // Agent-native Web3 operations
            Op::OracleRead => {
                if input_tensors.len() < 3 || input_tensors.len() > 4 {
                    return Err(VMError::WrongInputCount {
                        expected: 3,
                        got: input_tensors.len(),
                    });
                }
                let rpc_url = Self::require_string_scalar(input_tensors[0], "OracleRead rpc_url")?;
                let address = Self::require_string_scalar(input_tensors[1], "OracleRead address")?;
                let call_data = Self::require_string_scalar(input_tensors[2], "OracleRead call_data")?;
                let decimals = if let Some(tensor) = input_tensors.get(3) {
                    tensor.as_scalar().max(0.0) as u32
                } else {
                    0
                };
                let json = oracle_read_json(rpc_url, address, call_data, decimals)
                    .map_err(VMError::Web3Error)?;
                Ok(Tensor::scalar_string(
                    json,
                    Self::minimum_confidence(&input_tensors),
                ))
            }

            Op::GetGasPrice => {
                if input_tensors.is_empty() || input_tensors.len() > 3 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                let rpc_url = Self::require_string_scalar(input_tensors[0], "GetGasPrice rpc_url")?;
                let priority_fee_gwei = input_tensors.get(1).map(|tensor| tensor.as_scalar());
                let safety_multiplier = input_tensors.get(2).map(|tensor| tensor.as_scalar());
                let json = get_gas_price_json(rpc_url, priority_fee_gwei, safety_multiplier)
                    .map_err(VMError::Web3Error)?;
                Ok(Tensor::scalar_string(
                    json,
                    Self::minimum_confidence(&input_tensors),
                ))
            }

            // Confidence operations
            Op::ConfidenceCombine => {
                // Combine multiple confidence scores using geometric mean
                if input_tensors.is_empty() {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: 0,
                    });
                }
                self.execute_confidence_combine(&input_tensors)
            }

            Op::ConfidenceThreshold => {
                // Output 1.0 if confidence >= threshold, 0.0 otherwise
                // Inputs: [value_tensor, threshold_tensor]
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                let threshold = input_tensors[1].as_scalar();
                self.execute_confidence_threshold(input_tensors[0], threshold)
            }

            Op::ConfidenceDecay => {
                // Reduce confidence over time
                // Inputs: [value_tensor, decay_factor_tensor]
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                let decay_factor = input_tensors[1].as_scalar();
                self.execute_confidence_decay(input_tensors[0], decay_factor)
            }

            Op::ConfidenceBoost => {
                // Increase confidence based on input factor
                // Inputs: [value_tensor, boost_factor_tensor]
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                let boost_factor = input_tensors[1].as_scalar();
                self.execute_confidence_boost(input_tensors[0], boost_factor)
            }
        }
    }

    /// Combine multiple confidence scores using geometric mean
    fn execute_confidence_combine(&self, inputs: &[&Tensor]) -> Result<Tensor, VMError> {
        let confidences: Vec<f32> = inputs.iter().map(|t| t.confidence).collect();
        
        // Geometric mean: (product)^(1/n)
        let product: f32 = confidences.iter().product();
        let combined = product.powf(1.0 / confidences.len() as f32);
        
        // Return a scalar tensor with the combined confidence
        Ok(Tensor::confidence_scalar(combined))
    }

    /// Apply threshold to confidence: output 1.0 if >= threshold, else 0.0
    fn execute_confidence_threshold(
        &self,
        input: &Tensor,
        threshold: f32,
    ) -> Result<Tensor, VMError> {
        let passes = input.confidence >= threshold;
        let result_value = if passes { 1.0 } else { 0.0 };
        
        // Return tensor with same shape but thresholded confidence
        Ok(Tensor::scalar(result_value, input.confidence))
    }

    /// Decay confidence by a factor (0 to 1)
    fn execute_confidence_decay(
        &self,
        input: &Tensor,
        decay_factor: f32,
    ) -> Result<Tensor, VMError> {
        // Clamp decay factor to valid range
        let factor = decay_factor.clamp(0.0, 1.0);
        let new_confidence = input.confidence * factor;
        
        // Return tensor with decayed confidence
        Ok(Tensor::with_data(
            input.shape.clone(),
            input.data.clone(),
            new_confidence,
        ))
    }

    /// Boost confidence by a factor (clamped to max 1.0)
    fn execute_confidence_boost(
        &self,
        input: &Tensor,
        boost_factor: f32,
    ) -> Result<Tensor, VMError> {
        // Boost = confidence + (1 - confidence) * boost_factor
        // This approaches 1.0 asymptotically as boost_factor approaches 1.0
        let factor = boost_factor.clamp(0.0, 1.0);
        let new_confidence = (input.confidence + (1.0 - input.confidence) * factor).min(1.0);
        
        // Return tensor with boosted confidence
        Ok(Tensor::with_data(
            input.shape.clone(),
            input.data.clone(),
            new_confidence,
        ))
    }

    fn require_string_scalar<'a>(tensor: &'a Tensor, label: &str) -> Result<&'a str, VMError> {
        match &tensor.data {
            TensorData::String(values) if tensor.is_scalar() && !values.is_empty() => Ok(&values[0]),
            _ => Err(VMError::TensorError(format!(
                "{label} must be a scalar string tensor"
            ))),
        }
    }

    fn minimum_confidence(inputs: &[&Tensor]) -> f32 {
        inputs
            .iter()
            .map(|tensor| tensor.confidence)
            .fold(1.0f32, f32::min)
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
    /// Agent-native Web3 operation failed
    Web3Error(String),
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
            VMError::Web3Error(reason) => write!(f, "Web3 operation failed: {}", reason),
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
