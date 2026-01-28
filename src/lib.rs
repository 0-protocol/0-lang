//! ZeroLang - Agent-to-Agent Programming Language
//!
//! No syntax sugar. No whitespace. No variable names. Pure logic density.

pub mod tensor;
pub mod graph;
pub mod vm;
pub mod stdlib;

// Include the generated Cap'n Proto types
pub mod zero_capnp {
    include!(concat!(env!("OUT_DIR"), "/zero_capnp.rs"));
}

// Re-export commonly used types
pub use tensor::Tensor;
pub use graph::RuntimeGraph;
pub use vm::VM;

use sha2::{Sha256, Digest};

/// Compute SHA-256 hash of data
pub fn compute_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}
