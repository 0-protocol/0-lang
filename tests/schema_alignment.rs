//! Schema–Runtime Alignment Tests
//!
//! Verifies that every runtime node type and tensor data variant can survive
//! a Cap'n Proto serialize→deserialize round-trip, AND that old (float-only)
//! messages still parse correctly after the schema upgrade.

use capnp::message::{Builder, HeapAllocator};
use capnp::serialize;
use rust_decimal::Decimal;
use zerolang::zero_capnp::{graph, node, OverlapPolicy};
use zerolang::{RuntimeGraph, RuntimeNode};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hash(tag: u8) -> Vec<u8> {
    let mut h = vec![0u8; 32];
    h[0] = tag;
    h
}

fn round_trip(message: &Builder<HeapAllocator>) -> RuntimeGraph {
    let mut buf = Vec::new();
    serialize::write_message(&mut buf, message).expect("serialize");
    RuntimeGraph::from_reader(&buf[..]).expect("deserialize")
}

/// Build a minimal valid graph wrapper around a single node.
fn single_node_graph<F>(build_node: F) -> Builder<HeapAllocator>
where
    F: FnOnce(node::Builder, &[u8]),
{
    let mut message = Builder::new_default();
    let node_hash = hash(1);
    {
        let mut g = message.init_root::<graph::Builder>();
        g.set_version(0);

        let mut nodes = g.reborrow().init_nodes(1);
        {
            let mut nb = nodes.reborrow().get(0);
            nb.reborrow().init_id().set_hash(&node_hash);
            build_node(nb, &node_hash);
        }

        g.reborrow().init_entry_point().set_hash(&node_hash);
        {
            let mut outputs = g.reborrow().init_outputs(1);
            outputs.reborrow().get(0).set_hash(&node_hash);
        }
        {
            let mut proofs = g.reborrow().init_proofs(1);
            proofs.reborrow().get(0).init_halting().set_max_steps(1);
        }
    }
    message
}

// ===========================================================================
// 1. BACKWARD COMPATIBILITY – legacy float tensors
// ===========================================================================

#[test]
fn test_legacy_float_tensor_still_parses() {
    let msg = single_node_graph(|nb, _| {
        let mut t = nb.init_constant();
        t.reborrow().init_shape(1).set(0, 3);
        let mut data = t.reborrow().init_data(3);
        data.set(0, 1.0);
        data.set(1, 2.0);
        data.set(2, 3.0);
        t.set_confidence(0.95);
        // NOTE: no typedData set – this simulates the old format
    });

    let g = round_trip(&msg);
    let node = g.nodes.values().next().unwrap();
    match node {
        RuntimeNode::Constant(tensor) => {
            assert_eq!(tensor.shape, vec![3]);
            assert_eq!(*tensor.try_float_data().unwrap(), vec![1.0, 2.0, 3.0]);
            assert_eq!(tensor.confidence, 0.95);
        }
        other => panic!("Expected Constant, got {:?}", other),
    }
}

// ===========================================================================
// 2. TYPED TENSOR DATA – float via typedData
// ===========================================================================

#[test]
fn test_typed_float_data_round_trip() {
    let msg = single_node_graph(|nb, _| {
        let mut t = nb.init_constant();
        t.reborrow().init_shape(1).set(0, 2);
        t.set_confidence(1.0);
        {
            let td = t.reborrow().init_typed_data();
            let mut fd = td.init_float_data(2);
            fd.set(0, 42.0);
            fd.set(1, 99.0);
        }
    });

    let g = round_trip(&msg);
    let node = g.nodes.values().next().unwrap();
    match node {
        RuntimeNode::Constant(tensor) => {
            assert!(tensor.data.is_float());
            assert_eq!(*tensor.try_float_data().unwrap(), vec![42.0, 99.0]);
        }
        other => panic!("Expected Constant, got {:?}", other),
    }
}

// ===========================================================================
// 3. TYPED TENSOR DATA – string
// ===========================================================================

#[test]
fn test_typed_string_data_round_trip() {
    let msg = single_node_graph(|nb, _| {
        let mut t = nb.init_constant();
        t.reborrow().init_shape(1).set(0, 2);
        t.set_confidence(0.8);
        {
            let td = t.reborrow().init_typed_data();
            let mut sd = td.init_string_data(2);
            sd.set(0, "BTC/USD");
            sd.set(1, "ETH/USD");
        }
    });

    let g = round_trip(&msg);
    let node = g.nodes.values().next().unwrap();
    match node {
        RuntimeNode::Constant(tensor) => {
            assert!(tensor.data.is_string());
            let strings = tensor.data.as_string().unwrap();
            assert_eq!(strings, &["BTC/USD".to_string(), "ETH/USD".to_string()]);
            assert_eq!(tensor.confidence, 0.8);
        }
        other => panic!("Expected Constant, got {:?}", other),
    }
}

// ===========================================================================
// 4. TYPED TENSOR DATA – decimal
// ===========================================================================

#[test]
fn test_typed_decimal_data_round_trip() {
    let d1 = Decimal::new(12345, 2); // 123.45
    let d2 = Decimal::new(67890, 3); // 67.890

    let msg = single_node_graph(|nb, _| {
        let mut t = nb.init_constant();
        t.reborrow().init_shape(1).set(0, 2);
        t.set_confidence(0.99);
        {
            let td = t.reborrow().init_typed_data();
            let mut dd = td.init_decimal_data(2);
            dd.set(0, &d1.serialize());
            dd.set(1, &d2.serialize());
        }
    });

    let g = round_trip(&msg);
    let node = g.nodes.values().next().unwrap();
    match node {
        RuntimeNode::Constant(tensor) => {
            assert!(tensor.data.is_decimal());
            let decimals = tensor.data.as_decimal().unwrap();
            assert_eq!(decimals[0], d1);
            assert_eq!(decimals[1], d2);
        }
        other => panic!("Expected Constant, got {:?}", other),
    }
}

// ===========================================================================
// 5. TYPED TENSOR DATA – streamRef
// ===========================================================================

#[test]
fn test_typed_stream_ref_round_trip() {
    let msg = single_node_graph(|nb, _| {
        let mut t = nb.init_constant();
        t.reborrow().init_shape(1).set(0, 0);
        t.set_confidence(1.0);
        {
            let td = t.reborrow().init_typed_data();
            let mut sr = td.init_stream_ref();
            sr.set_id(42);
            sr.reborrow()
                .get_source_type()
                .set_websocket("wss://feed.example.com/v1");
        }
    });

    let g = round_trip(&msg);
    let node = g.nodes.values().next().unwrap();
    match node {
        RuntimeNode::Constant(tensor) => {
            assert!(tensor.data.is_stream());
            let handle = tensor.data.as_stream().unwrap();
            assert_eq!(handle.id, 42);
            match &handle.source_type {
                zerolang::StreamSource::WebSocket { url } => {
                    assert_eq!(url, "wss://feed.example.com/v1");
                }
                other => panic!("Expected WebSocket, got {:?}", other),
            }
        }
        other => panic!("Expected Constant, got {:?}", other),
    }
}

// ===========================================================================
// 6. NODE – permission
// ===========================================================================

#[test]
fn test_permission_node_round_trip() {
    let subject_hash = hash(10);
    let action_hash = hash(11);
    let fallback_hash = hash(12);

    let mut message = Builder::new_default();
    {
        let mut g = message.init_root::<graph::Builder>();
        g.set_version(0);

        let perm_hash = hash(1);
        let mut nodes = g.reborrow().init_nodes(4);

        // Dummy constant nodes for subject, action, fallback
        for (i, h) in [&subject_hash, &action_hash, &fallback_hash]
            .iter()
            .enumerate()
        {
            let mut nb = nodes.reborrow().get(i as u32);
            nb.reborrow().init_id().set_hash(h);
            let mut t = nb.init_constant();
            t.reborrow().init_shape(1).set(0, 1);
            t.reborrow().init_data(1).set(0, 1.0);
            t.set_confidence(1.0);
        }

        // Permission node
        {
            let mut nb = nodes.reborrow().get(3);
            nb.reborrow().init_id().set_hash(&perm_hash);
            let mut perm = nb.init_permission();
            perm.reborrow().init_subject().set_hash(&subject_hash);
            perm.reborrow().init_action().set_hash(&action_hash);
            perm.set_threshold(0.75);
            perm.reborrow().init_fallback().set_hash(&fallback_hash);
        }

        g.reborrow().init_entry_point().set_hash(&perm_hash);
        g.reborrow().init_outputs(1).get(0).set_hash(&perm_hash);
        g.reborrow().init_proofs(1).get(0).init_halting().set_max_steps(10);
    }

    let g = round_trip(&message);
    assert_eq!(g.nodes.len(), 4);

    let perm_node = g.nodes.get(&hash(1)).unwrap();
    match perm_node {
        RuntimeNode::Permission {
            subject,
            action,
            threshold,
            fallback,
        } => {
            assert_eq!(subject, &subject_hash);
            assert_eq!(action, &action_hash);
            assert_eq!(*threshold, 0.75);
            assert_eq!(fallback, &fallback_hash);
        }
        other => panic!("Expected Permission, got {:?}", other),
    }
}

// ===========================================================================
// 7. NODE – route
// ===========================================================================

#[test]
fn test_route_node_round_trip() {
    let input_hash = hash(10);
    let cond_hash = hash(11);
    let target_hash = hash(12);
    let default_hash = hash(13);
    let route_hash = hash(1);

    let mut message = Builder::new_default();
    {
        let mut g = message.init_root::<graph::Builder>();
        g.set_version(0);

        let mut nodes = g.reborrow().init_nodes(5);

        // Dummy constants
        for (i, h) in [&input_hash, &cond_hash, &target_hash, &default_hash]
            .iter()
            .enumerate()
        {
            let mut nb = nodes.reborrow().get(i as u32);
            nb.reborrow().init_id().set_hash(h);
            let mut t = nb.init_constant();
            t.reborrow().init_shape(1).set(0, 1);
            t.reborrow().init_data(1).set(0, 1.0);
            t.set_confidence(1.0);
        }

        // Route node
        {
            let mut nb = nodes.reborrow().get(4);
            nb.reborrow().init_id().set_hash(&route_hash);
            let mut rt = nb.init_route();
            rt.reborrow().init_input().set_hash(&input_hash);
            rt.reborrow().init_default().set_hash(&default_hash);

            let mut routes = rt.reborrow().init_routes(1);
            {
                let mut entry = routes.reborrow().get(0);
                entry.reborrow().init_condition().set_hash(&cond_hash);
                entry.set_threshold(0.9);
                entry.reborrow().init_target().set_hash(&target_hash);
                entry.set_name("primary");
                entry.set_description("Primary routing path");
            }
        }

        g.reborrow().init_entry_point().set_hash(&route_hash);
        g.reborrow().init_outputs(1).get(0).set_hash(&route_hash);
        g.reborrow().init_proofs(1).get(0).init_halting().set_max_steps(10);
    }

    let g = round_trip(&message);
    let route_node = g.nodes.get(&route_hash).unwrap();
    match route_node {
        RuntimeNode::Route {
            input,
            routes,
            default,
        } => {
            assert_eq!(input, &input_hash);
            assert_eq!(default, &default_hash);
            assert_eq!(routes.len(), 1);
            assert_eq!(routes[0].condition, cond_hash);
            assert_eq!(routes[0].threshold, 0.9);
            assert_eq!(routes[0].target, target_hash);
            assert_eq!(routes[0].metadata.name, "primary");
            assert_eq!(routes[0].metadata.description, "Primary routing path");
        }
        other => panic!("Expected Route, got {:?}", other),
    }
}

// ===========================================================================
// 8. NODE – timer
// ===========================================================================

#[test]
fn test_timer_node_round_trip() {
    let target_hash = hash(10);
    let timer_hash = hash(1);

    let mut message = Builder::new_default();
    {
        let mut g = message.init_root::<graph::Builder>();
        g.set_version(0);

        let mut nodes = g.reborrow().init_nodes(2);

        // Dummy target constant
        {
            let mut nb = nodes.reborrow().get(0);
            nb.reborrow().init_id().set_hash(&target_hash);
            let mut t = nb.init_constant();
            t.reborrow().init_shape(1).set(0, 1);
            t.reborrow().init_data(1).set(0, 0.0);
            t.set_confidence(1.0);
        }

        // Timer node
        {
            let mut nb = nodes.reborrow().get(1);
            nb.reborrow().init_id().set_hash(&timer_hash);
            let mut tm = nb.init_timer();
            tm.set_schedule("*/5 * * * *");
            tm.reborrow().init_target().set_hash(&target_hash);
            tm.set_max_concurrent(3);
            tm.set_overlap_policy(OverlapPolicy::Queue);
        }

        g.reborrow().init_entry_point().set_hash(&timer_hash);
        g.reborrow().init_outputs(1).get(0).set_hash(&timer_hash);
        g.reborrow().init_proofs(1).get(0).init_halting().set_max_steps(10);
    }

    let g = round_trip(&message);
    let timer_node = g.nodes.get(&timer_hash).unwrap();
    match timer_node {
        RuntimeNode::Timer {
            schedule,
            target,
            max_concurrent,
            overlap_policy,
        } => {
            assert_eq!(schedule, "*/5 * * * *");
            assert_eq!(target, &target_hash);
            assert_eq!(*max_concurrent, 3);
            assert_eq!(*overlap_policy, zerolang::OverlapPolicy::Queue);
        }
        other => panic!("Expected Timer, got {:?}", other),
    }
}

// ===========================================================================
// 9. BACKWARD COMPAT – old conformance graphs still work
// ===========================================================================

#[test]
fn test_existing_conformance_graphs_unaffected() {
    use zerolang::{stdlib, VM};

    let message = stdlib::generate_hello_world(0).unwrap();
    let mut buf = Vec::new();
    serialize::write_message(&mut buf, &message).unwrap();
    let g = RuntimeGraph::from_reader(&buf[..]).unwrap();

    let mut vm = VM::new();
    let outputs = vm.execute(&g).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![768]);

    let message = stdlib::generate_simple_math(0).unwrap();
    let mut buf = Vec::new();
    serialize::write_message(&mut buf, &message).unwrap();
    let g = RuntimeGraph::from_reader(&buf[..]).unwrap();

    let mut vm = VM::new();
    let outputs = vm.execute(&g).unwrap();
    assert_eq!(outputs[0].try_as_scalar().unwrap(), 3.0);
}
