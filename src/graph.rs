//! Graph - The DAG structure for ZeroLang programs
//!
//! Converts Cap'n Proto serialized graphs into executable in-memory structures.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use capnp::message::ReaderOptions;
use capnp::serialize;

use crate::zero_capnp::{graph, node, proof};
use crate::Tensor;

/// A hash-based node identifier (content-addressable)
pub type NodeHash = Vec<u8>;

/// Runtime representation of a node in the graph
#[derive(Debug, Clone)]
pub enum RuntimeNode {
    /// A constant tensor value
    Constant(Tensor),
    /// An operation with inputs
    Operation { op: Op, inputs: Vec<NodeHash> },
    /// A branch (probabilistic control flow)
    Branch {
        condition: NodeHash,
        threshold: f32,
        true_branch: NodeHash,
        false_branch: NodeHash,
    },
    /// External reference (FFI, other graphs)
    External { uri: String, inputs: Vec<NodeHash> },
}

/// Supported operations (matches schema/zero.capnp Operation enum)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    // Tensor math
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    // Activations
    Softmax,
    Relu,
    Sigmoid,
    Tanh,
    // Comparisons (output Tensor<1> confidence)
    Eq,
    Gt,
    Lt,
    // Reductions
    Sum,
    Mean,
    Argmax,
    // Shape manipulation
    Reshape,
    Transpose,
    Concat,
    // Special
    Identity,
    Embed,
}

impl Op {
    fn from_capnp(op: crate::zero_capnp::Operation) -> Result<Self, GraphError> {
        use crate::zero_capnp::Operation;
        match op {
            // Tensor math
            Operation::Add => Ok(Op::Add),
            Operation::Sub => Ok(Op::Sub),
            Operation::Mul => Ok(Op::Mul),
            Operation::Div => Ok(Op::Div),
            Operation::Matmul => Ok(Op::Matmul),
            // Activations
            Operation::Softmax => Ok(Op::Softmax),
            Operation::Relu => Ok(Op::Relu),
            Operation::Sigmoid => Ok(Op::Sigmoid),
            Operation::Tanh => Ok(Op::Tanh),
            // Comparisons
            Operation::Eq => Ok(Op::Eq),
            Operation::Gt => Ok(Op::Gt),
            Operation::Lt => Ok(Op::Lt),
            // Reductions
            Operation::Sum => Ok(Op::Sum),
            Operation::Mean => Ok(Op::Mean),
            Operation::Argmax => Ok(Op::Argmax),
            // Shape manipulation
            Operation::Reshape => Ok(Op::Reshape),
            Operation::Transpose => Ok(Op::Transpose),
            Operation::Concat => Ok(Op::Concat),
            // Special
            Operation::Identity => Ok(Op::Identity),
            Operation::Embed => Ok(Op::Embed),
        }
    }
}

/// Proof attached to a graph
#[derive(Debug, Clone)]
pub enum RuntimeProof {
    /// Halting proof - guarantees termination
    Halting { max_steps: u64, fuel_budget: u64 },
    /// Shape validity proof
    ShapeValid {
        input_shapes: Vec<Vec<u32>>,
        output_shape: Vec<u32>,
    },
    /// Cryptographic signature
    Signature {
        agent_id: Vec<u8>,
        signature: Vec<u8>,
        timestamp: u64,
    },
    /// No proof (unsafe)
    None,
}

/// The runtime graph structure
#[derive(Debug)]
pub struct RuntimeGraph {
    /// All nodes indexed by their hash
    pub nodes: HashMap<NodeHash, RuntimeNode>,
    /// The entry point hash
    pub entry_point: NodeHash,
    /// Output node hashes
    pub outputs: Vec<NodeHash>,
    /// Protocol version
    pub version: u16,
    /// Attached proofs
    pub proofs: Vec<RuntimeProof>,
}

impl RuntimeGraph {
    /// Load a graph from a .0 file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, GraphError> {
        let file = File::open(path).map_err(|e| GraphError::IoError(e.to_string()))?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Load a graph from a reader
    pub fn from_reader<R: std::io::BufRead>(reader: R) -> Result<Self, GraphError> {
        let message_reader = serialize::read_message(reader, ReaderOptions::new())
            .map_err(|e| GraphError::ParseError(e.to_string()))?;
        let graph_reader = message_reader
            .get_root::<graph::Reader>()
            .map_err(|e| GraphError::ParseError(e.to_string()))?;

        Self::from_capnp(graph_reader)
    }

    /// Parse a graph from Cap'n Proto reader
    fn from_capnp(reader: graph::Reader) -> Result<Self, GraphError> {
        let version = reader.get_version();

        // Parse entry point
        let entry_point = reader
            .get_entry_point()
            .map_err(|e| GraphError::ParseError(e.to_string()))?
            .get_hash()
            .map_err(|e| GraphError::ParseError(e.to_string()))?
            .to_vec();

        // Parse outputs
        let outputs_reader = reader
            .get_outputs()
            .map_err(|e| GraphError::ParseError(e.to_string()))?;
        let mut outputs = Vec::new();
        for output in outputs_reader.iter() {
            let hash = output
                .get_hash()
                .map_err(|e| GraphError::ParseError(e.to_string()))?
                .to_vec();
            outputs.push(hash);
        }

        // Parse nodes
        let nodes_reader = reader
            .get_nodes()
            .map_err(|e| GraphError::ParseError(e.to_string()))?;
        let mut nodes = HashMap::new();

        for node_reader in nodes_reader.iter() {
            let id = node_reader
                .get_id()
                .map_err(|e| GraphError::ParseError(e.to_string()))?;
            let hash = id
                .get_hash()
                .map_err(|e| GraphError::ParseError(e.to_string()))?
                .to_vec();

            let runtime_node = match node_reader
                .which()
                .map_err(|e| GraphError::ParseError(e.to_string()))?
            {
                node::Constant(tensor_reader) => {
                    let tensor =
                        tensor_reader.map_err(|e| GraphError::ParseError(e.to_string()))?;
                    let shape: Vec<u32> = tensor
                        .get_shape()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .iter()
                        .collect();
                    let data: Vec<f32> = tensor
                        .get_data()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .iter()
                        .collect();
                    let confidence = tensor.get_confidence();

                    RuntimeNode::Constant(Tensor::new(shape, data, confidence))
                }
                node::Operation(op_reader) => {
                    let op = Op::from_capnp(
                        op_reader
                            .get_op()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?,
                    )?;
                    let inputs_reader = op_reader
                        .get_inputs()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?;
                    let mut inputs = Vec::new();
                    for input in inputs_reader.iter() {
                        let input_hash = input
                            .get_hash()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_vec();
                        inputs.push(input_hash);
                    }
                    RuntimeNode::Operation { op, inputs }
                }
                node::Branch(br) => {
                    let condition = br
                        .get_condition()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();
                    let threshold = br.get_threshold();
                    let true_branch = br
                        .get_true_branch()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();
                    let false_branch = br
                        .get_false_branch()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();

                    RuntimeNode::Branch {
                        condition,
                        threshold,
                        true_branch,
                        false_branch,
                    }
                }
                node::External(ext) => {
                    let uri = ext
                        .get_uri()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_string()
                        .map_err(|e| GraphError::ParseError(format!("Invalid URI: {:?}", e)))?;
                    let input_mapping = ext
                        .get_input_mapping()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?;
                    let mut inputs = Vec::new();
                    for input in input_mapping.iter() {
                        let input_hash = input
                            .get_hash()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_vec();
                        inputs.push(input_hash);
                    }
                    RuntimeNode::External { uri, inputs }
                }
            };

            nodes.insert(hash, runtime_node);
        }

        // Parse proofs
        let proofs_reader = reader
            .get_proofs()
            .map_err(|e| GraphError::ParseError(e.to_string()))?;
        let mut proofs = Vec::new();

        for proof_reader in proofs_reader.iter() {
            let runtime_proof = match proof_reader
                .which()
                .map_err(|e| GraphError::ParseError(e.to_string()))?
            {
                proof::Halting(h) => RuntimeProof::Halting {
                    max_steps: h.get_max_steps(),
                    fuel_budget: h.get_fuel_budget(),
                },
                proof::ShapeValid(sv) => {
                    let mut input_shapes = Vec::new();
                    for shape in sv
                        .get_input_shapes()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .iter()
                    {
                        let shape_vec: Vec<u32> = shape
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .iter()
                            .collect();
                        input_shapes.push(shape_vec);
                    }
                    let output_shape: Vec<u32> = sv
                        .get_output_shape()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .iter()
                        .collect();
                    RuntimeProof::ShapeValid {
                        input_shapes,
                        output_shape,
                    }
                }
                proof::Signature(sig) => RuntimeProof::Signature {
                    agent_id: sig
                        .get_agent_id()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec(),
                    signature: sig
                        .get_sig()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec(),
                    timestamp: sig.get_timestamp(),
                },
                proof::None(()) => RuntimeProof::None,
            };
            proofs.push(runtime_proof);
        }

        Ok(RuntimeGraph {
            nodes,
            entry_point,
            outputs,
            version,
            proofs,
        })
    }

    /// Get a topological ordering of nodes for execution
    /// Returns nodes in an order where all dependencies come before their dependents
    pub fn topological_sort(&self) -> Result<Vec<NodeHash>, GraphError> {
        let mut visited: HashMap<NodeHash, bool> = HashMap::new();
        let mut result: Vec<NodeHash> = Vec::new();

        // Initialize all nodes as not visited
        for hash in self.nodes.keys() {
            visited.insert(hash.clone(), false);
        }

        // DFS from each output node
        for output in &self.outputs {
            self.topo_dfs(output, &mut visited, &mut result)?;
        }

        Ok(result)
    }

    fn topo_dfs(
        &self,
        hash: &NodeHash,
        visited: &mut HashMap<NodeHash, bool>,
        result: &mut Vec<NodeHash>,
    ) -> Result<(), GraphError> {
        if *visited.get(hash).unwrap_or(&false) {
            return Ok(());
        }

        visited.insert(hash.clone(), true);

        // Get dependencies and visit them first
        if let Some(node) = self.nodes.get(hash) {
            match node {
                RuntimeNode::Constant(_) => {
                    // No dependencies
                }
                RuntimeNode::Operation { inputs, .. } => {
                    for input in inputs {
                        self.topo_dfs(input, visited, result)?;
                    }
                }
                RuntimeNode::Branch {
                    condition,
                    true_branch,
                    false_branch,
                    ..
                } => {
                    self.topo_dfs(condition, visited, result)?;
                    self.topo_dfs(true_branch, visited, result)?;
                    self.topo_dfs(false_branch, visited, result)?;
                }
                RuntimeNode::External { inputs, .. } => {
                    for input in inputs {
                        self.topo_dfs(input, visited, result)?;
                    }
                }
            }
        } else {
            return Err(GraphError::NodeNotFound(hex::encode(hash)));
        }

        result.push(hash.clone());
        Ok(())
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the first halting proof if present
    pub fn get_halting_proof(&self) -> Option<(u64, u64)> {
        for proof in &self.proofs {
            if let RuntimeProof::Halting {
                max_steps,
                fuel_budget,
            } = proof
            {
                return Some((*max_steps, *fuel_budget));
            }
        }
        None
    }

    /// Check if the graph has any halting proof
    pub fn has_halting_proof(&self) -> bool {
        self.get_halting_proof().is_some()
    }
}

/// Graph errors
#[derive(Debug)]
pub enum GraphError {
    IoError(String),
    ParseError(String),
    NodeNotFound(String),
    UnsupportedOperation,
    UnsupportedNodeType(String),
    CycleDetected,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::IoError(e) => write!(f, "IO error: {}", e),
            GraphError::ParseError(e) => write!(f, "Parse error: {}", e),
            GraphError::NodeNotFound(h) => write!(f, "Node not found: {}", h),
            GraphError::UnsupportedOperation => write!(f, "Unsupported operation"),
            GraphError::UnsupportedNodeType(t) => write!(f, "Unsupported node type: {}", t),
            GraphError::CycleDetected => write!(f, "Cycle detected in graph"),
        }
    }
}

impl std::error::Error for GraphError {}
