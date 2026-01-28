//! Conformance Tests - Zero Entropy Verification
//!
//! These tests ensure that the core language behavior remains consistent.
//! Every PR must pass these tests to ensure no "human entropy" (ambiguity)
//! is introduced into the system.

use capnp::serialize;
use zerolang::{stdlib, verify_graph, RuntimeGraph, Tensor, VerifyOptions, VM};

// =============================================================================
// EMBEDDING / GENESIS TESTS
// =============================================================================

#[test]
fn test_hello_embedding_determinism() {
    let e1 = stdlib::hello_embedding();
    let e2 = stdlib::hello_embedding();
    assert_eq!(e1, e2);
}

#[test]
fn test_hello_world_conformance() {
    // 1. Generate the Golden Graph (Genesis Block)
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
    assert_eq!(
        outputs.len(),
        1,
        "Hello World must produce exactly 1 output"
    );

    let output = &outputs[0];
    assert_eq!(output.shape, vec![768], "Output must be a 768-dim vector");
    assert_eq!(output.confidence, 1.0, "Output confidence must be 1.0");

    // Verify content matches the canonical embedding
    let expected = stdlib::hello_embedding();
    assert_eq!(
        output.data, expected,
        "Output data must match the canonical embedding"
    );
}

// =============================================================================
// MATH / ARITHMETIC TESTS
// =============================================================================

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

// =============================================================================
// VERIFICATION TESTS
// =============================================================================

#[test]
fn test_verification_passes_for_valid_graph() {
    let message = stdlib::generate_hello_world(0).expect("Failed to generate graph");
    let mut buffer = Vec::new();
    serialize::write_message(&mut buffer, &message).expect("Failed to serialize");
    let graph = RuntimeGraph::from_reader(&buffer[..]).expect("Failed to load graph");

    let options = VerifyOptions {
        verify_hashes: true,
        require_halting_proof: true,
        verify_shape_proofs: false,
    };

    let result = verify_graph(&graph, &options);
    assert!(result.is_ok(), "Valid graph should pass verification");

    let result = result.unwrap();
    assert_eq!(result.nodes_verified, 1);
    assert!(result.halting_proof.is_some());
}

#[test]
fn test_verification_determinism() {
    // Verify the same graph twice produces identical results
    let message = stdlib::generate_simple_math(0).expect("Failed to generate graph");
    let mut buffer = Vec::new();
    serialize::write_message(&mut buffer, &message).expect("Failed to serialize");
    let graph = RuntimeGraph::from_reader(&buffer[..]).expect("Failed to load graph");

    let options = VerifyOptions {
        verify_hashes: true,
        require_halting_proof: false,
        verify_shape_proofs: false,
    };

    let result1 = verify_graph(&graph, &options).unwrap();
    let result2 = verify_graph(&graph, &options).unwrap();

    assert_eq!(result1.nodes_verified, result2.nodes_verified);
    assert_eq!(result1.hash_checks_passed, result2.hash_checks_passed);
}

// =============================================================================
// TENSOR OPERATION GOLDEN TESTS
// =============================================================================

#[test]
fn test_tensor_matmul_golden() {
    // [2,3] @ [3,2] = [2,2]
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0);
    let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 1.0);
    let c = a.matmul(&b).unwrap();

    // Golden values computed by hand:
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_tensor_softmax_golden() {
    let t = Tensor::from_vec(vec![0.0, 1.0, 2.0], 1.0);
    let s = t.softmax();

    // Verify properties of softmax
    let sum: f32 = s.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax must sum to 1.0");

    // All values must be positive
    assert!(s.data.iter().all(|&x| x > 0.0));

    // Monotonicity: higher input -> higher output
    assert!(s.data[2] > s.data[1]);
    assert!(s.data[1] > s.data[0]);
}

#[test]
fn test_tensor_confidence_propagation() {
    // Confidence should propagate as min(a, b) for binary ops
    let a = Tensor::scalar(1.0, 0.9);
    let b = Tensor::scalar(2.0, 0.7);
    let c = a.checked_add(&b).unwrap();

    assert_eq!(
        c.confidence, 0.7,
        "Confidence should be min(0.9, 0.7) = 0.7"
    );
}

// =============================================================================
// HASHING DETERMINISM TESTS
// =============================================================================

#[test]
fn test_canonical_hash_determinism() {
    use zerolang::verify::{hash_constant_node, hash_operation_node};
    use zerolang::Op;

    // Same tensor -> same hash
    let t1 = Tensor::scalar(42.0, 1.0);
    let t2 = Tensor::scalar(42.0, 1.0);
    assert_eq!(hash_constant_node(&t1), hash_constant_node(&t2));

    // Different tensor -> different hash
    let t3 = Tensor::scalar(43.0, 1.0);
    assert_ne!(hash_constant_node(&t1), hash_constant_node(&t3));

    // Same operation with same inputs -> same hash
    let h1 = hash_operation_node(Op::Add, &[vec![1; 32], vec![2; 32]]);
    let h2 = hash_operation_node(Op::Add, &[vec![1; 32], vec![2; 32]]);
    assert_eq!(h1, h2);

    // Different operation -> different hash
    let h3 = hash_operation_node(Op::Sub, &[vec![1; 32], vec![2; 32]]);
    assert_ne!(h1, h3);
}

#[test]
fn test_hash_changes_with_confidence() {
    use zerolang::verify::hash_constant_node;

    // Same value but different confidence -> different hash
    let t1 = Tensor::scalar(42.0, 1.0);
    let t2 = Tensor::scalar(42.0, 0.9);
    assert_ne!(hash_constant_node(&t1), hash_constant_node(&t2));
}
