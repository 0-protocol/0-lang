//! Conformance Tests - The Golden Rules of ZeroLang
//!
//! These tests ensure that the core language behavior remains consistent.
//! Every PR must pass these tests.

use zerolang::{stdlib, RuntimeGraph, VM};
use capnp::serialize;

#[test]
fn test_hello_embedding_determinism() {
    let e1 = stdlib::hello_embedding();
    let e2 = stdlib::hello_embedding();
    assert_eq!(e1, e2);
}

#[test]
fn test_hello_world_conformance() {
    // 1. Generate the Golden Graph (Genesis Block)
    // We use timestamp 0 to ensure binary determinism if we were checking the file hash,
    // but here we check behavioral conformance.
    let message = stdlib::generate_hello_world(0).expect("Failed to generate hello world graph");
    
    // 2. Serialize to buffer (mimicking file storage)
    let mut buffer = Vec::new();
    serialize::write_message(&mut buffer, &message).expect("Failed to serialize graph");
    
    // 3. Load into RuntimeGraph
    let graph = RuntimeGraph::from_reader(&buffer[..]).expect("Failed to load graph");
    
    // 4. Execute with VM
    let mut vm = VM::new();
    let outputs = vm.execute(&graph).expect("VM execution failed");
    
    // 5. Verify Golden Rules
    assert_eq!(outputs.len(), 1, "Hello World must produce exactly 1 output");
    
    let output = &outputs[0];
    assert_eq!(output.shape, vec![768], "Output must be a 768-dim vector");
    assert_eq!(output.confidence, 1.0, "Output confidence must be 1.0");
    
    // Verify content matches the canonical embedding
    let expected = stdlib::hello_embedding();
    assert_eq!(output.data, expected, "Output data must match the canonical embedding");
}

#[test]
fn test_simple_math_conformance() {
    // 1. Generate the Golden Graph (1.0 + 2.0 = 3.0)
    let message = stdlib::generate_simple_math(0).expect("Failed to generate math graph");
    
    // 2. Serialize
    let mut buffer = Vec::new();
    serialize::write_message(&mut buffer, &message).expect("Failed to serialize graph");
    
    // 3. Load
    let graph = RuntimeGraph::from_reader(&buffer[..]).expect("Failed to load graph");
    
    // 4. Execute
    let mut vm = VM::new();
    let outputs = vm.execute(&graph).expect("VM execution failed");
    
    // 5. Verify Golden Rules
    assert_eq!(outputs.len(), 1, "Math graph must produce exactly 1 output");
    
    let output = &outputs[0];
    assert_eq!(output.as_scalar(), 3.0, "1.0 + 2.0 must equal 3.0");
    assert_eq!(output.confidence, 1.0, "Math result confidence must be 1.0");
}
