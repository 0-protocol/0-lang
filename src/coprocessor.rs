//! Asynchronous ZK Coprocessor
//! Separates heavy ZK proof generation from the ultra-fast JIT VM execution loop.
//! "Optimistic Execution, Pessimistic Settlement"

use crate::vm::ExecutionTrace;
use std::future::Future;
use std::pin::Pin;

/// A trait for offloading execution traces to a hardware/software ZK prover.
pub trait ZkCoprocessor: Send + Sync {
    /// Takes an execution trace and asynchronously returns a cryptographic proof (e.g., SNARK/STARK).
    /// This does NOT block the main VM thread, allowing 1ms match times while proving takes 10s in the background.
    fn prove_trace_async(
        &self,
        trace: ExecutionTrace,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send>>;
}

/// A mock coprocessor for development and testing.
pub struct MockZkCoprocessor;

impl ZkCoprocessor for MockZkCoprocessor {
    fn prove_trace_async(
        &self,
        _trace: ExecutionTrace,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, String>> + Send>> {
        Box::pin(async move {
            // Simulate long ZK proving time without blocking the thread
            // tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            Ok(b"mock_zk_snark_proof_bytes".to_vec())
        })
    }
}
