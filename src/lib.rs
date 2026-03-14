//! ZeroLang - Agent-to-Agent Programming Language
//!
//! No syntax sugar. No whitespace. No variable names. Pure logic density.

pub mod events;
pub mod graph;
pub mod permission;
pub mod resolvers;
pub mod route;
pub mod stdlib;
pub mod stream;
pub mod tensor;
pub mod timer;
pub mod verify;
pub mod vm;
mod web3;

// Include the generated Cap'n Proto types
pub mod zero_capnp {
    include!(concat!(env!("OUT_DIR"), "/zero_capnp.rs"));
}

// Re-export commonly used types
pub use graph::{Op, OverlapPolicy, Route, RouteMetadata, RuntimeGraph, RuntimeNode, RuntimeProof};
pub use resolvers::{HttpMethod, HttpResolver, HttpResolverBuilder};
pub use tensor::{StreamHandle, StreamSource, Tensor, TensorData, TensorError};
pub use verify::{verify_graph, HaltingProofInfo, VerifyError, VerifyOptions, VerifyResult};
pub use vm::{ConfidenceCombineStrategy, ExecutionTrace, ExternalResolver, MockResolver, RejectingResolver, VMError, VM};

// Re-export events module
pub use events::{EventDispatcher, EventHandler, OrderStatus, SimpleEventHandler, TradingEvent};

// Re-export JSON utilities
pub use stdlib::json::{json_array, json_get, json_parse, JsonError};

// Re-export permission system (Agent #6 extension)
pub use permission::{
    CombinationStrategy, DefaultAction, PermissionAuditEntry, PermissionEvaluator,
    PermissionPolicy, PermissionResult, evaluate_permission,
};

// Re-export routing system (Agent #6 extension)
pub use route::{RouteBuilder, RouteConfig, RouteEvaluation, RouteResult, Router, RouterConfig, SelectedRoute};

// Re-export stream system (Agent #6 extension)
pub use stream::{StreamConfig, StreamError, StreamManager, StreamState};

// Re-export timer system (Agent #6 extension)
pub use timer::{
    TimerBuilder, TimerConfig, TimerError, TimerExecution, TimerManager, TimerState,
    schedules,
};

use sha2::{Digest, Sha256};

/// Compute SHA-256 hash of data
pub fn compute_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}
