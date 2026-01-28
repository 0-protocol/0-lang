//! Fuzz target for parsing and verifying .0 graph files
//!
//! This fuzz target ensures that:
//! 1. The parser doesn't panic on arbitrary input
//! 2. The verifier doesn't panic on parsed graphs
//! 3. All errors are properly handled via Result types

#![no_main]

use libfuzzer_sys::fuzz_target;
use zerolang::{verify_graph, RuntimeGraph, VerifyOptions, VM};

fuzz_target!(|data: &[u8]| {
    // Try to parse the data as a Zero graph
    // This should never panic - only return Ok or Err
    match RuntimeGraph::from_reader(data) {
        Ok(graph) => {
            // If parsing succeeds, try to verify the graph
            // This should also never panic
            let options = VerifyOptions {
                verify_hashes: true,
                require_halting_proof: false,
                verify_shape_proofs: false,
            };
            let _ = verify_graph(&graph, &options);

            // If verification passes, try to execute
            // This should also never panic
            let mut vm = VM::new();
            let _ = vm.execute(&graph);
        }
        Err(_) => {
            // Parse error is expected for most random inputs
            // Just ensure we didn't panic
        }
    }
});
