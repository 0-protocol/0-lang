//! Verification - Hash verification and graph validation for ZeroLang
//!
//! This module implements the canonical hashing specification and verification
//! logic for .0 files. Every node's hash should be deterministically computed
//! from its content.

use sha2::{Digest, Sha256};

use crate::graph::{GraphError, NodeHash, Op, RuntimeGraph, RuntimeNode};
use crate::Tensor;

/// Canonical hash prefix bytes for different node types.
/// These ensure different node types with same content produce different hashes.
const HASH_PREFIX_CONSTANT: u8 = 0x01;
const HASH_PREFIX_OPERATION: u8 = 0x02;
const HASH_PREFIX_BRANCH: u8 = 0x03;
const HASH_PREFIX_EXTERNAL: u8 = 0x04;

/// Compute the canonical hash of a Tensor.
///
/// Format: shape_len (u32 LE) + shape dims (u32 LE each) + data (f32 LE each) + confidence (f32 LE)
pub fn hash_tensor(tensor: &Tensor) -> Vec<u8> {
    let mut hasher = Sha256::new();

    // Shape length
    hasher.update((tensor.shape.len() as u32).to_le_bytes());

    // Shape dimensions
    for &dim in &tensor.shape {
        hasher.update(dim.to_le_bytes());
    }

    // Data
    for &val in &tensor.data {
        hasher.update(val.to_le_bytes());
    }

    // Confidence
    hasher.update(tensor.confidence.to_le_bytes());

    hasher.finalize().to_vec()
}

/// Compute the canonical hash of a Constant node.
///
/// Format: PREFIX_CONSTANT + tensor_hash
pub fn hash_constant_node(tensor: &Tensor) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update([HASH_PREFIX_CONSTANT]);
    hasher.update(hash_tensor(tensor));
    hasher.finalize().to_vec()
}

/// Compute the canonical hash of an Operation node.
///
/// Format: PREFIX_OPERATION + op_code (u8) + num_inputs (u32 LE) + input_hashes (in order)
pub fn hash_operation_node(op: Op, inputs: &[NodeHash]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update([HASH_PREFIX_OPERATION]);

    // Operation code
    hasher.update([op_to_byte(op)]);

    // Number of inputs
    hasher.update((inputs.len() as u32).to_le_bytes());

    // Input hashes in order
    for input in inputs {
        hasher.update(input);
    }

    hasher.finalize().to_vec()
}

/// Compute the canonical hash of a Branch node.
///
/// Format: PREFIX_BRANCH + condition_hash + threshold (f32 LE) + true_branch_hash + false_branch_hash
pub fn hash_branch_node(
    condition: &NodeHash,
    threshold: f32,
    true_branch: &NodeHash,
    false_branch: &NodeHash,
) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update([HASH_PREFIX_BRANCH]);
    hasher.update(condition);
    hasher.update(threshold.to_le_bytes());
    hasher.update(true_branch);
    hasher.update(false_branch);
    hasher.finalize().to_vec()
}

/// Compute the canonical hash of an External node.
///
/// Format: PREFIX_EXTERNAL + uri_len (u32 LE) + uri_bytes + num_inputs (u32 LE) + input_hashes
pub fn hash_external_node(uri: &str, inputs: &[NodeHash]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update([HASH_PREFIX_EXTERNAL]);

    // URI
    let uri_bytes = uri.as_bytes();
    hasher.update((uri_bytes.len() as u32).to_le_bytes());
    hasher.update(uri_bytes);

    // Number of inputs
    hasher.update((inputs.len() as u32).to_le_bytes());

    // Input hashes in order
    for input in inputs {
        hasher.update(input);
    }

    hasher.finalize().to_vec()
}

/// Compute the canonical hash of a RuntimeNode.
pub fn hash_node(node: &RuntimeNode) -> Vec<u8> {
    match node {
        RuntimeNode::Constant(tensor) => hash_constant_node(tensor),
        RuntimeNode::Operation { op, inputs } => hash_operation_node(*op, inputs),
        RuntimeNode::Branch {
            condition,
            threshold,
            true_branch,
            false_branch,
        } => hash_branch_node(condition, *threshold, true_branch, false_branch),
        RuntimeNode::External { uri, inputs } => hash_external_node(uri, inputs),
    }
}

/// Convert Op enum to a stable byte representation.
fn op_to_byte(op: Op) -> u8 {
    match op {
        Op::Add => 0,
        Op::Sub => 1,
        Op::Mul => 2,
        Op::Div => 3,
        Op::Matmul => 4,
        Op::Softmax => 5,
        Op::Relu => 6,
        Op::Sigmoid => 7,
        Op::Tanh => 8,
        Op::Eq => 9,
        Op::Gt => 10,
        Op::Lt => 11,
        Op::Sum => 12,
        Op::Mean => 13,
        Op::Argmax => 14,
        Op::Reshape => 15,
        Op::Transpose => 16,
        Op::Concat => 17,
        Op::Identity => 18,
        Op::Embed => 19,
        // New operations for trading
        Op::Gte => 20,
        Op::Lte => 21,
        Op::Min => 22,
        Op::Max => 23,
        Op::Abs => 24,
        Op::Neg => 25,
        Op::Clamp => 26,
    }
}

/// Verification errors
#[derive(Debug, Clone)]
pub enum VerifyError {
    /// A node's hash doesn't match its computed hash
    HashMismatch {
        node_hash: String,
        expected_hash: String,
    },
    /// A referenced node doesn't exist
    MissingNode(String),
    /// The graph contains a cycle
    CycleDetected,
    /// Entry point doesn't exist
    InvalidEntryPoint(String),
    /// Output node doesn't exist
    InvalidOutput(String),
    /// Graph is empty
    EmptyGraph,
    /// No halting proof present (in strict mode)
    MissingHaltingProof,
    /// Shape proof verification failed
    ShapeProofInvalid(String),
    /// Graph error during verification
    GraphError(String),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::HashMismatch {
                node_hash,
                expected_hash,
            } => {
                write!(
                    f,
                    "Hash mismatch: node claims {} but computed {}",
                    node_hash, expected_hash
                )
            }
            VerifyError::MissingNode(hash) => write!(f, "Missing node: {}", hash),
            VerifyError::CycleDetected => write!(f, "Cycle detected in graph"),
            VerifyError::InvalidEntryPoint(hash) => {
                write!(f, "Entry point not found: {}", hash)
            }
            VerifyError::InvalidOutput(hash) => write!(f, "Output node not found: {}", hash),
            VerifyError::EmptyGraph => write!(f, "Graph has no nodes"),
            VerifyError::MissingHaltingProof => {
                write!(f, "No halting proof present (required in strict mode)")
            }
            VerifyError::ShapeProofInvalid(reason) => {
                write!(f, "Shape proof invalid: {}", reason)
            }
            VerifyError::GraphError(e) => write!(f, "Graph error: {}", e),
        }
    }
}

impl std::error::Error for VerifyError {}

impl From<GraphError> for VerifyError {
    fn from(e: GraphError) -> Self {
        VerifyError::GraphError(e.to_string())
    }
}

/// Verification options
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    /// Verify node hashes match their content
    pub verify_hashes: bool,
    /// Require halting proof
    pub require_halting_proof: bool,
    /// Verify shape proofs
    pub verify_shape_proofs: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            verify_hashes: true,
            require_halting_proof: true,
            verify_shape_proofs: true,
        }
    }
}

impl VerifyOptions {
    /// Unsafe mode: skip all verification
    pub fn unsafe_mode() -> Self {
        Self {
            verify_hashes: false,
            require_halting_proof: false,
            verify_shape_proofs: false,
        }
    }
}

/// Verification result containing details about what was verified
#[derive(Debug)]
pub struct VerifyResult {
    /// Number of nodes verified
    pub nodes_verified: usize,
    /// Number of hash checks passed
    pub hash_checks_passed: usize,
    /// Halting proof details (if present)
    pub halting_proof: Option<HaltingProofInfo>,
}

/// Information extracted from halting proof
#[derive(Debug, Clone)]
pub struct HaltingProofInfo {
    pub max_steps: u64,
    pub fuel_budget: u64,
}

/// Verify a RuntimeGraph according to the given options.
pub fn verify_graph(
    graph: &RuntimeGraph,
    options: &VerifyOptions,
) -> Result<VerifyResult, VerifyError> {
    let mut result = VerifyResult {
        nodes_verified: 0,
        hash_checks_passed: 0,
        halting_proof: None,
    };

    // Check graph is not empty
    if graph.nodes.is_empty() {
        return Err(VerifyError::EmptyGraph);
    }

    // Verify entry point exists
    if !graph.nodes.contains_key(&graph.entry_point) {
        return Err(VerifyError::InvalidEntryPoint(hex::encode(
            &graph.entry_point,
        )));
    }

    // Verify all outputs exist
    for output in &graph.outputs {
        if !graph.nodes.contains_key(output) {
            return Err(VerifyError::InvalidOutput(hex::encode(output)));
        }
    }

    // Check halting proof requirement
    if let Some((max_steps, fuel_budget)) = graph.get_halting_proof() {
        result.halting_proof = Some(HaltingProofInfo {
            max_steps,
            fuel_budget,
        });
    } else if options.require_halting_proof {
        return Err(VerifyError::MissingHaltingProof);
    }

    // Verify each node
    for (hash, node) in &graph.nodes {
        result.nodes_verified += 1;

        // Verify hash matches content
        if options.verify_hashes {
            let computed_hash = hash_node(node);
            if &computed_hash != hash {
                return Err(VerifyError::HashMismatch {
                    node_hash: hex::encode(hash),
                    expected_hash: hex::encode(&computed_hash),
                });
            }
            result.hash_checks_passed += 1;
        }

        // Verify all referenced nodes exist
        match node {
            RuntimeNode::Constant(_) => {}
            RuntimeNode::Operation { inputs, .. } => {
                for input in inputs {
                    if !graph.nodes.contains_key(input) {
                        return Err(VerifyError::MissingNode(hex::encode(input)));
                    }
                }
            }
            RuntimeNode::Branch {
                condition,
                true_branch,
                false_branch,
                ..
            } => {
                if !graph.nodes.contains_key(condition) {
                    return Err(VerifyError::MissingNode(hex::encode(condition)));
                }
                if !graph.nodes.contains_key(true_branch) {
                    return Err(VerifyError::MissingNode(hex::encode(true_branch)));
                }
                if !graph.nodes.contains_key(false_branch) {
                    return Err(VerifyError::MissingNode(hex::encode(false_branch)));
                }
            }
            RuntimeNode::External { inputs, .. } => {
                for input in inputs {
                    if !graph.nodes.contains_key(input) {
                        return Err(VerifyError::MissingNode(hex::encode(input)));
                    }
                }
            }
        }
    }

    // Check for cycles using DFS
    verify_no_cycles(graph)?;

    Ok(result)
}

/// Verify the graph has no cycles using DFS with coloring.
fn verify_no_cycles(graph: &RuntimeGraph) -> Result<(), VerifyError> {
    use std::collections::HashMap;

    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White, // Not visited
        Gray,  // Currently visiting (in stack)
        Black, // Finished visiting
    }

    let mut colors: HashMap<&NodeHash, Color> = HashMap::new();
    for hash in graph.nodes.keys() {
        colors.insert(hash, Color::White);
    }

    fn dfs<'a>(
        hash: &'a NodeHash,
        graph: &'a RuntimeGraph,
        colors: &mut HashMap<&'a NodeHash, Color>,
    ) -> Result<(), VerifyError> {
        colors.insert(hash, Color::Gray);

        if let Some(node) = graph.nodes.get(hash) {
            let deps: Vec<&NodeHash> = match node {
                RuntimeNode::Constant(_) => vec![],
                RuntimeNode::Operation { inputs, .. } => inputs.iter().collect(),
                RuntimeNode::Branch {
                    condition,
                    true_branch,
                    false_branch,
                    ..
                } => vec![condition, true_branch, false_branch],
                RuntimeNode::External { inputs, .. } => inputs.iter().collect(),
            };

            for dep in deps {
                match colors.get(dep) {
                    Some(Color::Gray) => return Err(VerifyError::CycleDetected),
                    Some(Color::White) => dfs(dep, graph, colors)?,
                    _ => {}
                }
            }
        }

        colors.insert(hash, Color::Black);
        Ok(())
    }

    // Run DFS from all nodes
    for hash in graph.nodes.keys() {
        if colors.get(hash) == Some(&Color::White) {
            dfs(hash, graph, &mut colors)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RuntimeProof;
    use std::collections::HashMap;

    #[test]
    fn test_tensor_hash_determinism() {
        let t1 = Tensor::scalar(42.0, 1.0);
        let t2 = Tensor::scalar(42.0, 1.0);

        let h1 = hash_tensor(&t1);
        let h2 = hash_tensor(&t2);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_tensor_hash_different_values() {
        let t1 = Tensor::scalar(42.0, 1.0);
        let t2 = Tensor::scalar(43.0, 1.0);

        let h1 = hash_tensor(&t1);
        let h2 = hash_tensor(&t2);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_constant_node_hash() {
        let tensor = Tensor::scalar(1.0, 1.0);
        let hash1 = hash_constant_node(&tensor);
        let hash2 = hash_constant_node(&tensor);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_operation_node_hash_input_order_matters() {
        let hash_a = vec![1u8; 32];
        let hash_b = vec![2u8; 32];

        let h1 = hash_operation_node(Op::Add, &[hash_a.clone(), hash_b.clone()]);
        let h2 = hash_operation_node(Op::Add, &[hash_b.clone(), hash_a.clone()]);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_branch_node_hash() {
        let cond = vec![1u8; 32];
        let true_b = vec![2u8; 32];
        let false_b = vec![3u8; 32];

        let h1 = hash_branch_node(&cond, 0.5, &true_b, &false_b);
        let h2 = hash_branch_node(&cond, 0.5, &true_b, &false_b);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_external_node_hash() {
        let inputs = vec![vec![1u8; 32]];
        let h1 = hash_external_node("ffi:test", &inputs);
        let h2 = hash_external_node("ffi:test", &inputs);
        assert_eq!(h1, h2);

        let h3 = hash_external_node("ffi:other", &inputs);
        assert_ne!(h1, h3);
    }

    // Helper to create a simple valid graph
    fn make_simple_graph() -> RuntimeGraph {
        let tensor = Tensor::scalar(42.0, 1.0);
        let hash = hash_constant_node(&tensor);

        let mut nodes = HashMap::new();
        nodes.insert(hash.clone(), RuntimeNode::Constant(tensor));

        RuntimeGraph {
            nodes,
            entry_point: hash.clone(),
            outputs: vec![hash],
            version: 0,
            proofs: vec![],
        }
    }

    #[test]
    fn test_verify_simple_graph() {
        let graph = make_simple_graph();
        let options = VerifyOptions {
            verify_hashes: true,
            require_halting_proof: false,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.nodes_verified, 1);
        assert_eq!(result.hash_checks_passed, 1);
    }

    #[test]
    fn test_verify_empty_graph() {
        let graph = RuntimeGraph {
            nodes: HashMap::new(),
            entry_point: vec![1, 2, 3],
            outputs: vec![],
            version: 0,
            proofs: vec![],
        };
        let options = VerifyOptions {
            verify_hashes: true,
            require_halting_proof: false,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(matches!(result, Err(VerifyError::EmptyGraph)));
    }

    #[test]
    fn test_verify_missing_entry_point() {
        let tensor = Tensor::scalar(42.0, 1.0);
        let hash = hash_constant_node(&tensor);

        let mut nodes = HashMap::new();
        nodes.insert(hash.clone(), RuntimeNode::Constant(tensor));

        let graph = RuntimeGraph {
            nodes,
            entry_point: vec![0; 32], // Wrong hash
            outputs: vec![hash],
            version: 0,
            proofs: vec![],
        };
        let options = VerifyOptions {
            verify_hashes: false,
            require_halting_proof: false,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(matches!(result, Err(VerifyError::InvalidEntryPoint(_))));
    }

    #[test]
    fn test_verify_missing_output() {
        let tensor = Tensor::scalar(42.0, 1.0);
        let hash = hash_constant_node(&tensor);

        let mut nodes = HashMap::new();
        nodes.insert(hash.clone(), RuntimeNode::Constant(tensor));

        let graph = RuntimeGraph {
            nodes,
            entry_point: hash,
            outputs: vec![vec![0; 32]], // Wrong hash
            version: 0,
            proofs: vec![],
        };
        let options = VerifyOptions {
            verify_hashes: false,
            require_halting_proof: false,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(matches!(result, Err(VerifyError::InvalidOutput(_))));
    }

    #[test]
    fn test_verify_hash_mismatch() {
        let tensor = Tensor::scalar(42.0, 1.0);
        let _correct_hash = hash_constant_node(&tensor);
        let wrong_hash = vec![0; 32]; // Wrong hash

        let mut nodes = HashMap::new();
        nodes.insert(wrong_hash.clone(), RuntimeNode::Constant(tensor));

        let graph = RuntimeGraph {
            nodes,
            entry_point: wrong_hash.clone(),
            outputs: vec![wrong_hash],
            version: 0,
            proofs: vec![],
        };
        let options = VerifyOptions {
            verify_hashes: true,
            require_halting_proof: false,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(matches!(result, Err(VerifyError::HashMismatch { .. })));
    }

    #[test]
    fn test_verify_missing_node_reference() {
        let tensor_a = Tensor::scalar(1.0, 1.0);
        let hash_a = hash_constant_node(&tensor_a);
        let missing_hash = vec![0; 32];
        let op_hash = hash_operation_node(Op::Add, &[hash_a.clone(), missing_hash.clone()]);

        let mut nodes = HashMap::new();
        nodes.insert(hash_a.clone(), RuntimeNode::Constant(tensor_a));
        nodes.insert(
            op_hash.clone(),
            RuntimeNode::Operation {
                op: Op::Add,
                inputs: vec![hash_a, missing_hash],
            },
        );

        let graph = RuntimeGraph {
            nodes,
            entry_point: op_hash.clone(),
            outputs: vec![op_hash],
            version: 0,
            proofs: vec![],
        };
        let options = VerifyOptions {
            verify_hashes: false,
            require_halting_proof: false,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(matches!(result, Err(VerifyError::MissingNode(_))));
    }

    #[test]
    fn test_verify_cycle_detection() {
        // Create a cycle: A -> B -> A
        let hash_a = vec![1; 32];
        let hash_b = vec![2; 32];

        let mut nodes = HashMap::new();
        nodes.insert(
            hash_a.clone(),
            RuntimeNode::Operation {
                op: Op::Identity,
                inputs: vec![hash_b.clone()],
            },
        );
        nodes.insert(
            hash_b.clone(),
            RuntimeNode::Operation {
                op: Op::Identity,
                inputs: vec![hash_a.clone()],
            },
        );

        let graph = RuntimeGraph {
            nodes,
            entry_point: hash_a.clone(),
            outputs: vec![hash_a],
            version: 0,
            proofs: vec![],
        };
        let options = VerifyOptions {
            verify_hashes: false,
            require_halting_proof: false,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(matches!(result, Err(VerifyError::CycleDetected)));
    }

    #[test]
    fn test_verify_acyclic_diamond() {
        // Diamond pattern: A -> B, A -> C, B -> D, C -> D (valid, no cycle)
        let tensor_a = Tensor::scalar(1.0, 1.0);
        let hash_a = hash_constant_node(&tensor_a);

        let hash_b = hash_operation_node(Op::Identity, &[hash_a.clone()]);
        let hash_c = hash_operation_node(Op::Relu, &[hash_a.clone()]);
        let hash_d = hash_operation_node(Op::Add, &[hash_b.clone(), hash_c.clone()]);

        let mut nodes = HashMap::new();
        nodes.insert(hash_a.clone(), RuntimeNode::Constant(tensor_a));
        nodes.insert(
            hash_b.clone(),
            RuntimeNode::Operation {
                op: Op::Identity,
                inputs: vec![hash_a.clone()],
            },
        );
        nodes.insert(
            hash_c.clone(),
            RuntimeNode::Operation {
                op: Op::Relu,
                inputs: vec![hash_a.clone()],
            },
        );
        nodes.insert(
            hash_d.clone(),
            RuntimeNode::Operation {
                op: Op::Add,
                inputs: vec![hash_b, hash_c],
            },
        );

        let graph = RuntimeGraph {
            nodes,
            entry_point: hash_a,
            outputs: vec![hash_d],
            version: 0,
            proofs: vec![],
        };
        let options = VerifyOptions {
            verify_hashes: true,
            require_halting_proof: false,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_halting_proof_required() {
        let graph = make_simple_graph();
        let options = VerifyOptions {
            verify_hashes: false,
            require_halting_proof: true,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(matches!(result, Err(VerifyError::MissingHaltingProof)));
    }

    #[test]
    fn test_verify_with_halting_proof() {
        let tensor = Tensor::scalar(42.0, 1.0);
        let hash = hash_constant_node(&tensor);

        let mut nodes = HashMap::new();
        nodes.insert(hash.clone(), RuntimeNode::Constant(tensor));

        let graph = RuntimeGraph {
            nodes,
            entry_point: hash.clone(),
            outputs: vec![hash],
            version: 0,
            proofs: vec![RuntimeProof::Halting {
                max_steps: 100,
                fuel_budget: 1000,
            }],
        };
        let options = VerifyOptions {
            verify_hashes: true,
            require_halting_proof: true,
            verify_shape_proofs: false,
        };
        let result = verify_graph(&graph, &options);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.halting_proof.is_some());
        let hp = result.halting_proof.unwrap();
        assert_eq!(hp.max_steps, 100);
        assert_eq!(hp.fuel_budget, 1000);
    }
}
