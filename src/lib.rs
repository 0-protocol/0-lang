//! ZeroLang - Agent-to-Agent Programming Language
//!
//! No syntax sugar. No whitespace. No variable names. Pure logic density.

pub mod graph;
pub mod resolvers;
pub mod stdlib;
pub mod tensor;
pub mod verify;
pub mod vm;

// Include the generated Cap'n Proto types
pub mod zero_capnp {
    include!(concat!(env!("OUT_DIR"), "/zero_capnp.rs"));
}

// Re-export commonly used types
pub use graph::{Op, RuntimeGraph, RuntimeNode, RuntimeProof};
pub use resolvers::{HttpMethod, HttpResolver, HttpResolverBuilder};
pub use tensor::Tensor;
pub use verify::{verify_graph, HaltingProofInfo, VerifyError, VerifyOptions, VerifyResult};
pub use vm::{ExternalResolver, MockResolver, RejectingResolver, VMError, VM};

use sha2::{Digest, Sha256};

/// Compute SHA-256 hash of data
pub fn compute_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}
