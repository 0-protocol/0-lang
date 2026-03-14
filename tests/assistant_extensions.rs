//! Tests for Agent #6 Assistant Extensions
//!
//! These tests verify the new features added for AI assistant applications:
//! - StreamTensor
//! - PermissionNode
//! - RouteNode
//! - TimerNode
//! - Confidence operations
//! - Execution trace generation

use std::collections::HashMap;
use zerolang::{
    graph::{NodeHash, Op, Route, RouteMetadata, RuntimeNode},
    permission::{PermissionPolicy, CombinationStrategy, evaluate_permission},
    route::RouteBuilder,
    stream::StreamManager,
    tensor::Tensor,
    timer::{TimerManager, TimerBuilder, schedules},
    vm::VM,
    RuntimeGraph,
};

// ============================================================================
// StreamTensor Tests
// ============================================================================

#[tokio::test]
async fn test_stream_tensor_creation() {
    let mut manager = StreamManager::new();
    let handle = manager.from_websocket("wss://api.example.com/stream").await.unwrap();
    
    let tensor = Tensor::from_stream(handle.clone(), 1.0);
    assert!(tensor.data.is_stream());
    
    let stream = tensor.data.as_stream().unwrap();
    assert_eq!(stream.id, handle.id);
}

#[tokio::test]
async fn test_stream_manager_operations() {
    let mut manager = StreamManager::new();
    
    // Create multiple streams
    let ws_handle = manager.from_websocket("wss://example.com").await.unwrap();
    let channel_handle = manager.from_channel("test-channel").await.unwrap();
    let _event_handle = manager.from_event_source("message").await.unwrap();
    
    assert_eq!(manager.stream_count(), 3);
    
    // Push and read
    manager.push(&channel_handle, Tensor::scalar(42.0, 1.0)).await.unwrap();
    let result = manager.read(&channel_handle).await;
    assert!(result.is_some());
    assert_eq!(result.unwrap().as_scalar(), 42.0);
    
    // Close stream
    manager.close(ws_handle.id).await.unwrap();
}

// ============================================================================
// Permission System Tests
// ============================================================================

#[test]
fn test_permission_evaluation_basic() {
    let subject = Tensor::scalar(1.0, 0.9);
    let action = Tensor::scalar(1.0, 0.85);
    
    let policy = PermissionPolicy::with_threshold(0.8);
    let result = evaluate_permission(&subject, &action, &policy);
    
    assert!(result.allowed);
    assert!(result.confidence >= 0.8);
}

#[test]
fn test_permission_evaluation_denied() {
    let subject = Tensor::scalar(1.0, 0.6);
    let action = Tensor::scalar(1.0, 0.5);
    
    let policy = PermissionPolicy::with_threshold(0.8);
    let result = evaluate_permission(&subject, &action, &policy);
    
    assert!(!result.allowed);
}

#[test]
fn test_permission_combination_strategies() {
    let subject = Tensor::scalar(1.0, 0.9);
    let action = Tensor::scalar(1.0, 0.5);
    
    // Any strategy - should pass because 0.9 >= 0.7
    let policy_any = PermissionPolicy {
        threshold: 0.7,
        combination: CombinationStrategy::Any,
        audit: false,
        ..Default::default()
    };
    let result_any = evaluate_permission(&subject, &action, &policy_any);
    assert!(result_any.allowed);
    
    // All strategy - should fail because min(0.9, 0.5) = 0.5 < 0.7
    let policy_all = PermissionPolicy {
        threshold: 0.7,
        combination: CombinationStrategy::All,
        audit: false,
        ..Default::default()
    };
    let result_all = evaluate_permission(&subject, &action, &policy_all);
    assert!(!result_all.allowed);
}

// ============================================================================
// Routing System Tests
// ============================================================================

#[test]
fn test_route_selection() {
    let router = RouteBuilder::new()
        .route("high_priority", 0.9, vec![1])
        .route("medium_priority", 0.7, vec![2])
        .route("low_priority", 0.5, vec![3])
        .default(vec![0])
        .build();
    
    // Should select medium_priority (0.8 >= 0.7, but 0.8 < 0.9)
    let result = router.evaluate(&[0.5, 0.8, 0.6]);
    assert!(!result.used_default);
    assert_eq!(result.selected_route.unwrap().name, "medium_priority");
}

#[test]
fn test_route_default_selection() {
    let router = RouteBuilder::new()
        .route("high_only", 0.95, vec![1])
        .default(vec![0])
        .build();
    
    // Should use default because no route matches
    let result = router.evaluate(&[0.5]);
    assert!(result.used_default);
}

#[test]
fn test_route_highest_confidence() {
    let router = RouteBuilder::new()
        .route("first", 0.5, vec![1])
        .route("second", 0.5, vec![2])
        .route("third", 0.5, vec![3])
        .default(vec![0])
        .use_highest_confidence(true)
        .build();
    
    // Should select "third" with highest confidence 0.95
    let result = router.evaluate(&[0.6, 0.7, 0.95]);
    assert_eq!(result.selected_route.unwrap().name, "third");
}

// ============================================================================
// Timer System Tests
// ============================================================================

#[tokio::test]
async fn test_timer_registration() {
    let manager = TimerManager::new();
    let config = TimerBuilder::new(schedules::EVERY_5_MINUTES, vec![1, 2, 3])
        .name("test-timer")
        .max_concurrent(2)
        .build();
    
    let timer_id = manager.register(config).await.unwrap();
    let state = manager.get_state(timer_id).await.unwrap();
    
    assert_eq!(state, zerolang::timer::TimerState::Active);
}

#[tokio::test]
async fn test_timer_execution_tracking() {
    let manager = TimerManager::new();
    let config = TimerBuilder::new("*/5 * * * *", vec![1])
        .name("exec-test")
        .build();
    
    let timer_id = manager.register(config).await.unwrap();
    let exec_id = manager.trigger(timer_id).await.unwrap();
    
    manager.complete(exec_id, true, None).await.unwrap();
    
    let history = manager.get_history(timer_id).await;
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].success, Some(true));
}

// ============================================================================
// Confidence Operations Tests
// ============================================================================

#[test]
fn test_confidence_operations_in_vm() {
    // Create a simple graph with confidence operations
    let mut nodes = HashMap::new();
    
    // Constants with different confidences
    let hash1: NodeHash = vec![1];
    let hash2: NodeHash = vec![2];
    let hash3: NodeHash = vec![3]; // combine result
    
    nodes.insert(hash1.clone(), RuntimeNode::Constant(Tensor::scalar(1.0, 0.8)));
    nodes.insert(hash2.clone(), RuntimeNode::Constant(Tensor::scalar(2.0, 0.6)));
    nodes.insert(hash3.clone(), RuntimeNode::Operation {
        op: Op::ConfidenceCombine,
        inputs: vec![hash1.clone(), hash2.clone()],
    });
    
    let graph = RuntimeGraph {
        nodes,
        entry_point: hash1,
        outputs: vec![hash3],
        version: 1,
        proofs: vec![],
    };
    
    let mut vm = VM::new();
    let outputs = vm.execute(&graph).unwrap();
    
    // Geometric mean of 0.8 and 0.6 ≈ 0.693
    let confidence = outputs[0].confidence;
    assert!(confidence > 0.68 && confidence < 0.70);
}

#[test]
fn test_confidence_threshold_operation() {
    let mut nodes = HashMap::new();
    
    let value_hash: NodeHash = vec![1];
    let threshold_hash: NodeHash = vec![2];
    let result_hash: NodeHash = vec![3];
    
    nodes.insert(value_hash.clone(), RuntimeNode::Constant(Tensor::scalar(1.0, 0.85)));
    nodes.insert(threshold_hash.clone(), RuntimeNode::Constant(Tensor::scalar(0.8, 1.0)));
    nodes.insert(result_hash.clone(), RuntimeNode::Operation {
        op: Op::ConfidenceThreshold,
        inputs: vec![value_hash.clone(), threshold_hash.clone()],
    });
    
    let graph = RuntimeGraph {
        nodes,
        entry_point: value_hash,
        outputs: vec![result_hash],
        version: 1,
        proofs: vec![],
    };
    
    let mut vm = VM::new();
    let outputs = vm.execute(&graph).unwrap();
    
    // 0.85 >= 0.8, so should return 1.0
    assert_eq!(outputs[0].as_scalar(), 1.0);
}

// ============================================================================
// Execution Trace Tests
// ============================================================================

#[test]
fn test_execution_trace_generation() {
    let mut nodes = HashMap::new();
    
    let const1: NodeHash = vec![1];
    let const2: NodeHash = vec![2];
    let add_result: NodeHash = vec![3];
    
    nodes.insert(const1.clone(), RuntimeNode::Constant(Tensor::scalar(10.0, 0.9)));
    nodes.insert(const2.clone(), RuntimeNode::Constant(Tensor::scalar(20.0, 0.8)));
    nodes.insert(add_result.clone(), RuntimeNode::Operation {
        op: Op::Add,
        inputs: vec![const1.clone(), const2.clone()],
    });
    
    let graph = RuntimeGraph {
        nodes,
        entry_point: const1,
        outputs: vec![add_result],
        version: 1,
        proofs: vec![],
    };
    
    let mut vm = VM::new();
    let (outputs, trace) = vm.execute_with_trace(&graph).unwrap();
    
    assert_eq!(outputs[0].as_scalar(), 30.0);
    assert_eq!(trace.len(), 3); // 2 constants + 1 operation
    assert!(!trace.timestamps.is_empty());
    assert!(!trace.confidences.is_empty());
}

#[test]
fn test_execution_trace_confidence_tracking() {
    let mut nodes = HashMap::new();
    
    let const1: NodeHash = vec![1];
    nodes.insert(const1.clone(), RuntimeNode::Constant(Tensor::scalar(1.0, 0.75)));
    
    let graph = RuntimeGraph {
        nodes,
        entry_point: const1.clone(),
        outputs: vec![const1],
        version: 1,
        proofs: vec![],
    };
    
    let mut vm = VM::new();
    let (_, trace) = vm.execute_with_trace(&graph).unwrap();
    
    assert_eq!(trace.min_confidence(), 0.75);
    assert_eq!(trace.avg_confidence(), 0.75);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_permission_node_integration() {
    let mut nodes = HashMap::new();
    
    let subject_hash: NodeHash = vec![1];
    let action_hash: NodeHash = vec![2];
    let fallback_hash: NodeHash = vec![3];
    let permission_hash: NodeHash = vec![4];
    
    nodes.insert(subject_hash.clone(), RuntimeNode::Constant(Tensor::scalar(1.0, 0.9)));
    nodes.insert(action_hash.clone(), RuntimeNode::Constant(Tensor::scalar(1.0, 0.85)));
    nodes.insert(fallback_hash.clone(), RuntimeNode::Constant(Tensor::scalar(0.0, 1.0)));
    nodes.insert(permission_hash.clone(), RuntimeNode::Permission {
        subject: subject_hash.clone(),
        action: action_hash.clone(),
        threshold: 0.8,
        fallback: fallback_hash.clone(),
    });
    
    let graph = RuntimeGraph {
        nodes,
        entry_point: subject_hash,
        outputs: vec![permission_hash],
        version: 1,
        proofs: vec![],
    };
    
    let mut vm = VM::new();
    let outputs = vm.execute(&graph).unwrap();
    
    // Permission should be granted (min(0.9, 0.85) = 0.85 >= 0.8)
    assert!(outputs[0].confidence >= 0.8);
}

#[test]
fn test_route_node_integration() {
    let mut nodes = HashMap::new();
    
    let input_hash: NodeHash = vec![1];
    let cond1_hash: NodeHash = vec![2];
    let cond2_hash: NodeHash = vec![3];
    let target1_hash: NodeHash = vec![4];
    let target2_hash: NodeHash = vec![5];
    let default_hash: NodeHash = vec![6];
    let route_hash: NodeHash = vec![7];
    
    nodes.insert(input_hash.clone(), RuntimeNode::Constant(Tensor::scalar(1.0, 1.0)));
    nodes.insert(cond1_hash.clone(), RuntimeNode::Constant(Tensor::scalar(0.5, 0.5))); // Won't match
    nodes.insert(cond2_hash.clone(), RuntimeNode::Constant(Tensor::scalar(1.0, 0.8))); // Will match
    nodes.insert(target1_hash.clone(), RuntimeNode::Constant(Tensor::scalar(100.0, 1.0)));
    nodes.insert(target2_hash.clone(), RuntimeNode::Constant(Tensor::scalar(200.0, 1.0)));
    nodes.insert(default_hash.clone(), RuntimeNode::Constant(Tensor::scalar(0.0, 1.0)));
    
    let routes = vec![
        Route {
            condition: cond1_hash.clone(),
            threshold: 0.7,
            target: target1_hash.clone(),
            metadata: RouteMetadata {
                name: "route1".to_string(),
                description: "First route".to_string(),
            },
        },
        Route {
            condition: cond2_hash.clone(),
            threshold: 0.7,
            target: target2_hash.clone(),
            metadata: RouteMetadata {
                name: "route2".to_string(),
                description: "Second route".to_string(),
            },
        },
    ];
    
    nodes.insert(route_hash.clone(), RuntimeNode::Route {
        input: input_hash.clone(),
        routes,
        default: default_hash.clone(),
    });
    
    let graph = RuntimeGraph {
        nodes,
        entry_point: input_hash,
        outputs: vec![route_hash],
        version: 1,
        proofs: vec![],
    };
    
    let mut vm = VM::new();
    let outputs = vm.execute(&graph).unwrap();
    
    // Should select route2 (target2) because cond2 has confidence 0.8 >= 0.7
    assert_eq!(outputs[0].as_scalar(), 200.0);
}
