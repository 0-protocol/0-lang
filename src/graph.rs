//! Graph - The DAG structure for ZeroLang programs
//!
//! Converts Cap'n Proto serialized graphs into executable in-memory structures.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use capnp::message::ReaderOptions;
use capnp::serialize;

use crate::zero_capnp::{graph, node, proof, stream_ref, tensor, tensor_payload};
use crate::tensor::{StreamHandle, StreamSource, TensorData};
use crate::Tensor;

/// Three-color DFS markers for topological sort cycle detection.
const DFS_WHITE: u8 = 0; // Unvisited
const DFS_GRAY: u8 = 1;  // In current DFS path (back-edge target → cycle)
const DFS_BLACK: u8 = 2; // Fully processed

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
    /// Mutable state across executions (positions, balances)
    State { key: String, default: Tensor },
    /// Permission check with confidence threshold
    Permission {
        /// Who is requesting (subject tensor)
        subject: NodeHash,
        /// What they want to do (action tensor)
        action: NodeHash,
        /// Minimum confidence required to allow
        threshold: f32,
        /// What to execute if denied
        fallback: NodeHash,
    },
    /// Multi-path routing with priority ordering
    Route {
        /// Input to route
        input: NodeHash,
        /// Routes in priority order (first matching wins)
        routes: Vec<Route>,
        /// Default route if none match
        default: NodeHash,
    },
    /// Scheduled execution (cron-like)
    Timer {
        /// Cron expression (e.g., "*/5 * * * *")
        schedule: String,
        /// Graph to execute when timer fires
        target: NodeHash,
        /// Maximum concurrent executions
        max_concurrent: u32,
        /// What to do if execution is still running
        overlap_policy: OverlapPolicy,
    },
}

/// A single route in a RouteNode
#[derive(Debug, Clone)]
pub struct Route {
    /// Condition graph (must output confidence > threshold)
    pub condition: NodeHash,
    /// Minimum confidence to take this route
    pub threshold: f32,
    /// Target node if route matches
    pub target: NodeHash,
    /// Route metadata (for tracing)
    pub metadata: RouteMetadata,
}

/// Metadata for route tracing and debugging
#[derive(Debug, Clone)]
pub struct RouteMetadata {
    /// Human-readable route name
    pub name: String,
    /// Route description
    pub description: String,
}

/// Policy for handling overlapping timer executions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlapPolicy {
    /// Skip this execution if previous is still running
    Skip,
    /// Queue execution for later
    Queue,
    /// Run in parallel
    Parallel,
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
    Gte,  // Greater than or equal
    Lte,  // Less than or equal
    // Reductions
    Sum,
    Mean,
    Argmax,
    Min,  // Element-wise or reduction min
    Max,  // Element-wise or reduction max
    // Shape manipulation
    Reshape,
    Transpose,
    Concat,
    // Special
    Identity,
    Embed,
    Abs,    // Absolute value
    Neg,    // Negation
    Clamp,  // Clamp values to range (useful for position limits)
    // JSON operations (for API responses)
    JsonParse,  // Parse JSON string into structured tensor
    JsonGet,    // Extract value by key path (e.g., "data.price")
    JsonArray,  // Extract array elements
    // Agent-native Web3 operations
    OracleRead,   // Read oracle/contract state via EVM eth_call
    GetGasPrice,  // Fetch EVM gas price quote
    VerifySignature,
    GetBlockDrift,    // Fetch block time drift vs current timestamp
    StateChannelSign, // Evaluate and sign matched ASTs
    ExtractASTHash,   // Hash the structural AST skeleton

    // Phase 5: Sentient Compilation
    ForkState,        // Snapshot VM state into a portable hash
    MergeState,       // Hydrate VM state from a hash
    EmbedDistance,    // Semantic cosine similarity between two embeddings
    MutateAST,        // Adaptively output a mutated graph blueprint

    // Phase 8: The Synaptic Interface
    VerifyCognition,  // Validate pre-computed, cryptographically signed LLM reasoning
    StreamIngest,     // Multimodal pointer resolution (video/audio/text streams)

    // Confidence operations (for AI assistant applications)
    ConfidenceCombine,    // Combine multiple confidence scores
    ConfidenceThreshold,  // Output 1.0 if above threshold, 0.0 otherwise
    ConfidenceDecay,      // Reduce confidence over time
    ConfidenceBoost,      // Increase confidence based on history
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
            Operation::Gte => Ok(Op::Gte),
            Operation::Lte => Ok(Op::Lte),
            // Reductions
            Operation::Sum => Ok(Op::Sum),
            Operation::Mean => Ok(Op::Mean),
            Operation::Argmax => Ok(Op::Argmax),
            Operation::Min => Ok(Op::Min),
            Operation::Max => Ok(Op::Max),
            // Shape manipulation
            Operation::Reshape => Ok(Op::Reshape),
            Operation::Transpose => Ok(Op::Transpose),
            Operation::Concat => Ok(Op::Concat),
            // Special
            Operation::Identity => Ok(Op::Identity),
            Operation::Embed => Ok(Op::Embed),
            // Math operations (for trading)
            Operation::Abs => Ok(Op::Abs),
            Operation::Neg => Ok(Op::Neg),
            Operation::Clamp => Ok(Op::Clamp),
            // JSON operations
            Operation::JsonParse => Ok(Op::JsonParse),
            Operation::JsonGet => Ok(Op::JsonGet),
            Operation::JsonArray => Ok(Op::JsonArray),
            // Agent-native Web3 operations
            Operation::OracleRead => Ok(Op::OracleRead),
            Operation::GetGasPrice => Ok(Op::GetGasPrice),
            Operation::VerifySignature => Ok(Op::VerifySignature),
            Operation::GetBlockDrift => Ok(Op::GetBlockDrift),
            Operation::StateChannelSign => Ok(Op::StateChannelSign),
            Operation::ExtractAstHash => Ok(Op::ExtractASTHash),
            Operation::ForkState => Ok(Op::ForkState),
            Operation::MergeState => Ok(Op::MergeState),
            Operation::EmbedDistance => Ok(Op::EmbedDistance),
            Operation::MutateAst => Ok(Op::MutateAST),
            Operation::VerifyCognition => Ok(Op::VerifyCognition),
            Operation::StreamIngest => Ok(Op::StreamIngest),
            // Confidence operations
            Operation::ConfidenceCombine => Ok(Op::ConfidenceCombine),
            Operation::ConfidenceThreshold => Ok(Op::ConfidenceThreshold),
            Operation::ConfidenceDecay => Ok(Op::ConfidenceDecay),
            Operation::ConfidenceBoost => Ok(Op::ConfidenceBoost),
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

/// Deserialize a Tensor from its Cap'n Proto representation.
///
/// Backward compatibility: if `typedData` is present, it determines the data type;
/// otherwise we fall back to the legacy `data @1` float list.
fn tensor_from_capnp(reader: tensor::Reader) -> Result<Tensor, GraphError> {
    let shape: Vec<u32> = reader
        .get_shape()
        .map_err(|e| GraphError::ParseError(e.to_string()))?
        .iter()
        .collect();
    let confidence = reader.get_confidence();

    if reader.has_typed_data() {
        let typed = reader
            .get_typed_data()
            .map_err(|e| GraphError::ParseError(e.to_string()))?;
        match typed
            .which()
            .map_err(|e| GraphError::ParseError(e.to_string()))?
        {
            tensor_payload::FloatData(data) => {
                let data: Vec<f32> = data
                    .map_err(|e| GraphError::ParseError(e.to_string()))?
                    .iter()
                    .collect();
                Ok(Tensor::new(shape, data, confidence))
            }
            tensor_payload::StringData(list) => {
                let list = list.map_err(|e| GraphError::ParseError(e.to_string()))?;
                let mut strings = Vec::with_capacity(list.len() as usize);
                for i in 0..list.len() {
                    let s = list
                        .get(i)
                        .map_err(|e| GraphError::ParseError(e.to_string()))?;
                    let s = s
                        .to_str()
                        .map_err(|e| GraphError::ParseError(format!("Invalid UTF-8: {}", e)))?;
                    strings.push(s.to_owned());
                }
                Ok(Tensor::with_data(
                    shape,
                    TensorData::String(strings),
                    confidence,
                ))
            }
            tensor_payload::DecimalData(list) => {
                let list = list.map_err(|e| GraphError::ParseError(e.to_string()))?;
                let mut decimals = Vec::with_capacity(list.len() as usize);
                for i in 0..list.len() {
                    let bytes = list
                        .get(i)
                        .map_err(|e| GraphError::ParseError(e.to_string()))?;
                    if bytes.len() != 16 {
                        return Err(GraphError::ParseError(format!(
                            "Decimal data must be 16 bytes, got {}",
                            bytes.len()
                        )));
                    }
                    let mut buf = [0u8; 16];
                    buf.copy_from_slice(bytes);
                    decimals.push(rust_decimal::Decimal::deserialize(buf));
                }
                Ok(Tensor::with_data(
                    shape,
                    TensorData::Decimal(decimals),
                    confidence,
                ))
            }
            tensor_payload::StreamRef(ref_reader) => {
                let ref_reader =
                    ref_reader.map_err(|e| GraphError::ParseError(e.to_string()))?;
                let id = ref_reader.get_id();
                let source_type = match ref_reader
                    .get_source_type()
                    .which()
                    .map_err(|e| GraphError::ParseError(e.to_string()))?
                {
                    stream_ref::source_type::Websocket(url) => {
                        let url = url
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_str()
                            .map_err(|e| {
                                GraphError::ParseError(format!("Invalid UTF-8: {}", e))
                            })?
                            .to_owned();
                        StreamSource::WebSocket { url }
                    }
                    stream_ref::source_type::Channel(ch) => {
                        let channel_id = ch
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_str()
                            .map_err(|e| {
                                GraphError::ParseError(format!("Invalid UTF-8: {}", e))
                            })?
                            .to_owned();
                        StreamSource::Channel { channel_id }
                    }
                    stream_ref::source_type::Event(ev) => {
                        let event_type = ev
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_str()
                            .map_err(|e| {
                                GraphError::ParseError(format!("Invalid UTF-8: {}", e))
                            })?
                            .to_owned();
                        StreamSource::Event { event_type }
                    }
                };
                let handle = StreamHandle {
                    id,
                    source_type,
                    buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
                        std::collections::VecDeque::new(),
                    )),
                };
                Ok(Tensor::from_stream(handle, confidence))
            }
        }
    } else {
        // Legacy path: read float data from the original `data @1` field
        let data: Vec<f32> = reader
            .get_data()
            .map_err(|e| GraphError::ParseError(e.to_string()))?
            .iter()
            .collect();
        Ok(Tensor::new(shape, data, confidence))
    }
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
                    let tensor_r =
                        tensor_reader.map_err(|e| GraphError::ParseError(e.to_string()))?;
                    RuntimeNode::Constant(tensor_from_capnp(tensor_r)?)
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
                node::State(st) => {
                    let key = st
                        .get_key()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_string()
                        .map_err(|e| GraphError::ParseError(format!("Invalid key: {:?}", e)))?;
                    let default_tensor =
                        st.get_default().map_err(|e| GraphError::ParseError(e.to_string()))?;
                    let default = tensor_from_capnp(default_tensor)?;
                    RuntimeNode::State { key, default }
                }
                node::Permission(perm) => {
                    let subject = perm
                        .get_subject()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();
                    let action = perm
                        .get_action()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();
                    let threshold = perm.get_threshold();
                    let fallback = perm
                        .get_fallback()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();
                    RuntimeNode::Permission {
                        subject,
                        action,
                        threshold,
                        fallback,
                    }
                }
                node::Route(rt) => {
                    let input = rt
                        .get_input()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();
                    let routes_reader = rt
                        .get_routes()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?;
                    let mut routes = Vec::new();
                    for entry in routes_reader.iter() {
                        let condition = entry
                            .get_condition()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .get_hash()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_vec();
                        let threshold = entry.get_threshold();
                        let target = entry
                            .get_target()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .get_hash()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_vec();
                        let name = entry
                            .get_name()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_str()
                            .map_err(|e| {
                                GraphError::ParseError(format!("Invalid UTF-8: {}", e))
                            })?
                            .to_owned();
                        let description = entry
                            .get_description()
                            .map_err(|e| GraphError::ParseError(e.to_string()))?
                            .to_str()
                            .map_err(|e| {
                                GraphError::ParseError(format!("Invalid UTF-8: {}", e))
                            })?
                            .to_owned();
                        routes.push(Route {
                            condition,
                            threshold,
                            target,
                            metadata: RouteMetadata { name, description },
                        });
                    }
                    let default = rt
                        .get_default()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();
                    RuntimeNode::Route {
                        input,
                        routes,
                        default,
                    }
                }
                node::Timer(tm) => {
                    let schedule = tm
                        .get_schedule()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_str()
                        .map_err(|e| GraphError::ParseError(format!("Invalid UTF-8: {}", e)))?
                        .to_owned();
                    let target = tm
                        .get_target()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .get_hash()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                        .to_vec();
                    let max_concurrent = tm.get_max_concurrent();
                    let overlap_policy = match tm
                        .get_overlap_policy()
                        .map_err(|e| GraphError::ParseError(e.to_string()))?
                    {
                        crate::zero_capnp::OverlapPolicy::Skip => OverlapPolicy::Skip,
                        crate::zero_capnp::OverlapPolicy::Queue => OverlapPolicy::Queue,
                        crate::zero_capnp::OverlapPolicy::Parallel => OverlapPolicy::Parallel,
                    };
                    RuntimeNode::Timer {
                        schedule,
                        target,
                        max_concurrent,
                        overlap_policy,
                    }
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

    /// Get a topological ordering of nodes for execution.
    /// Returns nodes in an order where all dependencies come before their dependents.
    /// Returns `GraphError::CycleDetected` if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<NodeHash>, GraphError> {
        let mut colors: HashMap<NodeHash, u8> = HashMap::new();
        let mut result: Vec<NodeHash> = Vec::new();

        for hash in self.nodes.keys() {
            colors.insert(hash.clone(), DFS_WHITE);
        }

        for output in &self.outputs {
            Self::topo_dfs_impl(output, &self.nodes, &mut colors, &mut result)?;
        }

        Ok(result)
    }

    fn topo_dfs_impl(
        hash: &NodeHash,
        nodes: &HashMap<NodeHash, RuntimeNode>,
        colors: &mut HashMap<NodeHash, u8>,
        result: &mut Vec<NodeHash>,
    ) -> Result<(), GraphError> {
        match colors.get(hash).copied() {
            Some(DFS_BLACK) => return Ok(()),
            Some(DFS_GRAY) => return Err(GraphError::CycleDetected),
            _ => {}
        }

        colors.insert(hash.clone(), DFS_GRAY);

        if let Some(node) = nodes.get(hash) {
            let deps: Vec<&NodeHash> = match node {
                RuntimeNode::Constant(_) => vec![],
                RuntimeNode::State { .. } => vec![],
                RuntimeNode::Operation { inputs, .. } => inputs.iter().collect(),
                RuntimeNode::Branch {
                    condition,
                    true_branch,
                    false_branch,
                    ..
                } => vec![condition, true_branch, false_branch],
                RuntimeNode::External { inputs, .. } => inputs.iter().collect(),
                RuntimeNode::Permission {
                    subject,
                    action,
                    fallback,
                    ..
                } => vec![subject, action, fallback],
                RuntimeNode::Route {
                    input,
                    routes,
                    default,
                } => {
                    let mut deps = vec![input, default];
                    for route in routes {
                        deps.push(&route.condition);
                        deps.push(&route.target);
                    }
                    deps
                }
                RuntimeNode::Timer { target, .. } => vec![target],
            };

            for dep in deps {
                Self::topo_dfs_impl(dep, nodes, colors, result)?;
            }
        } else {
            return Err(GraphError::NodeNotFound(hex::encode(hash)));
        }

        colors.insert(hash.clone(), DFS_BLACK);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::{hash_constant_node, hash_operation_node};

    fn make_linear_graph() -> RuntimeGraph {
        let tensor_a = Tensor::scalar(1.0, 1.0);
        let hash_a = hash_constant_node(&tensor_a);
        let hash_b = hash_operation_node(Op::Identity, std::slice::from_ref(&hash_a));
        let hash_c = hash_operation_node(Op::Relu, std::slice::from_ref(&hash_b));

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
                inputs: vec![hash_b],
            },
        );

        RuntimeGraph {
            nodes,
            entry_point: hash_a,
            outputs: vec![hash_c],
            version: 0,
            proofs: vec![],
        }
    }

    #[test]
    fn test_topo_sort_cycle_detected() {
        let hash_a = vec![1u8; 32];
        let hash_b = vec![2u8; 32];

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

        let result = graph.topological_sort();
        assert!(
            matches!(result, Err(GraphError::CycleDetected)),
            "expected CycleDetected, got {:?}",
            result
        );
    }

    #[test]
    fn test_topo_sort_self_loop_detected() {
        let hash_a = vec![3u8; 32];

        let mut nodes = HashMap::new();
        nodes.insert(
            hash_a.clone(),
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

        let result = graph.topological_sort();
        assert!(
            matches!(result, Err(GraphError::CycleDetected)),
            "expected CycleDetected for self-loop, got {:?}",
            result
        );
    }

    #[test]
    fn test_topo_sort_linear_order() {
        let graph = make_linear_graph();
        let order = graph.topological_sort().expect("acyclic graph should sort");

        assert_eq!(order.len(), 3);
        // Dependencies must appear before dependents
        let pos = |h: &NodeHash| order.iter().position(|x| x == h).unwrap();

        for (hash, node) in &graph.nodes {
            if let RuntimeNode::Operation { inputs, .. } = node {
                for input in inputs {
                    assert!(
                        pos(input) < pos(hash),
                        "dependency should precede dependent"
                    );
                }
            }
        }
    }

    #[test]
    fn test_topo_sort_diamond_no_cycle() {
        let tensor = Tensor::scalar(1.0, 1.0);
        let hash_a = hash_constant_node(&tensor);
        let hash_b = hash_operation_node(Op::Identity, std::slice::from_ref(&hash_a));
        let hash_c = hash_operation_node(Op::Relu, std::slice::from_ref(&hash_a));
        let hash_d = hash_operation_node(Op::Add, &[hash_b.clone(), hash_c.clone()]);

        let mut nodes = HashMap::new();
        nodes.insert(hash_a.clone(), RuntimeNode::Constant(tensor));
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
                inputs: vec![hash_b.clone(), hash_c.clone()],
            },
        );

        let graph = RuntimeGraph {
            nodes,
            entry_point: hash_a,
            outputs: vec![hash_d.clone()],
            version: 0,
            proofs: vec![],
        };

        let order = graph.topological_sort().expect("diamond is acyclic");
        assert_eq!(order.len(), 4);

        let pos = |h: &NodeHash| order.iter().position(|x| x == h).unwrap();
        assert!(pos(&hash_b) < pos(&hash_d));
        assert!(pos(&hash_c) < pos(&hash_d));
    }
}

impl RuntimeGraph {
    /// 🛡️ AST Genetic Resonance (Structural Entropy Extractor)
    /// Hashes ONLY the structural layout (opcodes/connections) while ignoring dynamic runtime scalar/tensor values.
    /// 🛡️ AST Genetic Resonance (Structural Entropy Extractor)
    /// Hashes the structural layout (opcodes/connections) PLUS the agent's identity and nonce.
    /// This prevents high-frequency market makers from being falsely flagged as spam when updating prices.

    /// 🛡️ Canonical IR Normalization
    /// Sorts inputs of commutative operations (Add, Mul, Max, Min, Eq) by their connection hashes
    /// to ensure mathematically equivalent ASTs produce identical structural hashes,
    /// preventing $2^N$ isomorphic Sybil attacks.
    fn sort_commutative_inputs(op: &Op, inputs: &mut Vec<(NodeHash, u32)>) {
        match op {
            Op::Add | Op::Mul | Op::Max | Op::Min | Op::Eq => {
                inputs.sort_by(|a, b| a.0.cmp(&b.0));
            },
            _ => {} // Non-commutative, preserve order
        }
    }

    pub fn structural_hash_with_identity(&self, agent_pubkey: &[u8], nonce: u64) -> [u8; 32] {
        use sha3::{Digest, Keccak256};
        let mut hasher = Keccak256::new();
        hasher.update(agent_pubkey);
        hasher.update(&nonce.to_le_bytes());
        
        let mut sorted_hashes: Vec<_> = self.nodes.keys().collect();
        sorted_hashes.sort(); // Ensure deterministic traversal
        
        for hash in sorted_hashes {
            if let Some(node) = self.nodes.get(hash) {
                match node {
                    RuntimeNode::Operation { op, inputs } => {
                        let mut sorted_inputs = inputs.clone();
                        Self::sort_commutative_inputs(op, &mut sorted_inputs);
                        
                        hasher.update(&[0x01]); // Op marker
                        hasher.update(format!("{:?}", op).as_bytes()); // Deterministic AST operation string
                        hasher.update(&(sorted_inputs.len() as u32).to_le_bytes());
                        for input in sorted_inputs {
                            hasher.update(&input.0);
                        }
                    },
                    RuntimeNode::Constant { output_shape, .. } => {
                        hasher.update(&[0x02]); // Constant marker
                        hasher.update(&(output_shape.len() as u32).to_le_bytes());
                    },
                    RuntimeNode::Input { name, .. } => {
                        hasher.update(&[0x03]); // Input marker
                        hasher.update(name.as_bytes());
                    }
                }
            }
        }
        let mut result = [0u8; 32];
        result.copy_from_slice(&hasher.finalize());
        result
    }
    
    // Legacy fallback for backward compatibility
    pub fn structural_hash(&self) -> [u8; 32] {
        use sha3::{Digest, Keccak256};
        let mut hasher = Keccak256::new();
        
        let mut sorted_hashes: Vec<_> = self.nodes.keys().collect();
        sorted_hashes.sort(); // Ensure deterministic traversal
        
        for hash in sorted_hashes {
            if let Some(node) = self.nodes.get(hash) {
                match node {
                    RuntimeNode::Operation { op: _, inputs } => {
                        hasher.update(&[0x01]); // Op marker
                        hasher.update(format!("{:?}", op).as_bytes()); // Deterministic AST operation string
                        hasher.update(&(inputs.len() as u32).to_le_bytes());
                        for input in inputs {
                            hasher.update(&input.0);
                        }
                    },
                    RuntimeNode::Constant { output_shape, .. } => {
                        hasher.update(&[0x02]); // Constant marker
                        hasher.update(&(output_shape.len() as u32).to_le_bytes());
                    },
                    RuntimeNode::Input { name, .. } => {
                        hasher.update(&[0x03]); // Input marker
                        hasher.update(name.as_bytes());
                    }
                }
            }
        }
        let mut result = [0u8; 32];
        result.copy_from_slice(&hasher.finalize());
        result
    }
}
