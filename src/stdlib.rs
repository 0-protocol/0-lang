//! Standard Library - Golden Rule Graph Generators
//!
//! This module contains the canonical implementations of the standard
//! 0-lang graphs used for verification and testing.

use capnp::message::{Builder, HeapAllocator};
use crate::zero_capnp::{graph, Operation};
use crate::{compute_hash, Tensor};

/// The "Hello" concept encoded as a 768-dimensional embedding vector.
pub fn hello_embedding() -> Vec<f32> {
    let mut embedding = vec![0.0f32; 768];
    
    // Encode "HELLO" as ASCII values normalized to [0, 1]
    embedding[0] = 72.0 / 255.0;  // H
    embedding[1] = 69.0 / 255.0;  // E
    embedding[2] = 76.0 / 255.0;  // L
    embedding[3] = 76.0 / 255.0;  // L
    embedding[4] = 79.0 / 255.0;  // O
    
    // Fill remaining dimensions with a recognizable pattern
    for i in 5..768 {
        embedding[i] = ((i as f32) * 0.1).sin() * 0.5 + 0.5;
    }
    
    embedding
}

/// Generate the "Hello World" graph - the Genesis Block of ZeroLang
pub fn generate_hello_world(timestamp: u64) -> Result<Builder<HeapAllocator>, Box<dyn std::error::Error>> {
    let mut message = Builder::new_default();
    
    {
        let mut graph_builder = message.init_root::<graph::Builder>();
        graph_builder.set_version(0);
        
        let embedding = hello_embedding();
        let embedding_bytes: Vec<u8> = embedding.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let node_hash = compute_hash(&embedding_bytes);
        
        let mut nodes = graph_builder.reborrow().init_nodes(1);
        
        {
            let mut node_builder = nodes.reborrow().get(0);
            {
                let mut id_builder = node_builder.reborrow().init_id();
                id_builder.set_hash(&node_hash);
            }
            {
                let mut tensor_builder = node_builder.init_constant();
                let mut shape = tensor_builder.reborrow().init_shape(1);
                shape.set(0, 768);
                let mut data = tensor_builder.reborrow().init_data(768);
                for (i, &val) in embedding.iter().enumerate() {
                    data.set(i as u32, val);
                }
                tensor_builder.set_confidence(1.0);
            }
        }
        
        {
            let mut entry = graph_builder.reborrow().init_entry_point();
            entry.set_hash(&node_hash);
        }
        {
            let mut outputs = graph_builder.reborrow().init_outputs(1);
            outputs.reborrow().get(0).set_hash(&node_hash);
        }
        {
            let mut proofs = graph_builder.reborrow().init_proofs(1);
            let proof_builder = proofs.reborrow().get(0);
            let mut halting = proof_builder.init_halting();
            halting.set_max_steps(1);
            halting.set_fuel_budget(1);
        }
        {
            let mut metadata = graph_builder.reborrow().get_metadata();
            metadata.set_created_by(b"zerolang-genesis");
            metadata.set_created_at(timestamp);
            metadata.set_description(b"The Genesis Block - Hello World in ZeroLang");
        }
    }
    
    Ok(message)
}

/// Generate a simple math graph: 1.0 + 2.0 = 3.0
pub fn generate_simple_math(timestamp: u64) -> Result<Builder<HeapAllocator>, Box<dyn std::error::Error>> {
    let mut message = Builder::new_default();
    
    {
        let mut graph_builder = message.init_root::<graph::Builder>();
        graph_builder.set_version(0);
        
        // Create tensors for hashing
        let tensor_a = Tensor::scalar(1.0, 1.0);
        let tensor_b = Tensor::scalar(2.0, 1.0);
        
        let hash_a = compute_hash(&tensor_a.to_bytes());
        let hash_b = compute_hash(&tensor_b.to_bytes());
        
        // Hash for the operation node (hash of operation + inputs)
        let mut op_content = vec![0u8]; // Operation::Add = 0
        op_content.extend(&hash_a);
        op_content.extend(&hash_b);
        let hash_result = compute_hash(&op_content);
        
        // Create 3 nodes: A=1.0, B=2.0, Result=A+B
        let mut nodes = graph_builder.reborrow().init_nodes(3);
        
        // Node A: Constant 1.0
        {
            let mut node_builder = nodes.reborrow().get(0);
            node_builder.reborrow().init_id().set_hash(&hash_a);
            let mut tensor_builder = node_builder.init_constant();
            tensor_builder.reborrow().init_shape(1).set(0, 1);
            tensor_builder.reborrow().init_data(1).set(0, 1.0);
            tensor_builder.set_confidence(1.0);
        }
        
        // Node B: Constant 2.0
        {
            let mut node_builder = nodes.reborrow().get(1);
            node_builder.reborrow().init_id().set_hash(&hash_b);
            let mut tensor_builder = node_builder.init_constant();
            tensor_builder.reborrow().init_shape(1).set(0, 1);
            tensor_builder.reborrow().init_data(1).set(0, 2.0);
            tensor_builder.set_confidence(1.0);
        }
        
        // Node Result: A + B
        {
            let mut node_builder = nodes.reborrow().get(2);
            node_builder.reborrow().init_id().set_hash(&hash_result);
            let mut op_builder = node_builder.init_operation();
            op_builder.set_op(Operation::Add);
            let mut inputs = op_builder.reborrow().init_inputs(2);
            inputs.reborrow().get(0).set_hash(&hash_a);
            inputs.reborrow().get(1).set_hash(&hash_b);
        }
        
        // Entry point is first constant
        graph_builder.reborrow().init_entry_point().set_hash(&hash_a);
        
        // Output is the result node
        {
            let mut outputs = graph_builder.reborrow().init_outputs(1);
            outputs.reborrow().get(0).set_hash(&hash_result);
        }
        
        // Halting proof
        {
            let mut proofs = graph_builder.reborrow().init_proofs(1);
            let proof_builder = proofs.reborrow().get(0);
            let mut halting = proof_builder.init_halting();
            halting.set_max_steps(3);
            halting.set_fuel_budget(1);
        }
        
        // Metadata
        {
            let mut metadata = graph_builder.reborrow().get_metadata();
            metadata.set_created_by(b"zerolang-math");
            metadata.set_created_at(timestamp);
            metadata.set_description(b"Simple Math: 1.0 + 2.0 = 3.0");
        }
    }
    
    Ok(message)
}
