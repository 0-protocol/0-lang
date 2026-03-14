use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use serde_json::json;
use zerolang::{Op, RuntimeGraph, RuntimeNode, Tensor, VM};

fn spawn_mock_rpc_server(responses: Vec<serde_json::Value>) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{}", addr);

    let handle = thread::spawn(move || {
        for response_json in responses {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = [0u8; 8192];
            let _ = stream.read(&mut buffer).unwrap();

            let body = serde_json::to_string(&response_json).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        }
    });

    (url, handle)
}

#[test]
fn test_get_gas_price_can_feed_json_get() {
    let gas_price = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": "0x2540be400"
    });
    let latest_block = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "baseFeePerGas": "0x4a817c800"
        }
    });
    let priority_fee = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": "0x77359400"
    });
    let (rpc_url, server) = spawn_mock_rpc_server(vec![gas_price, latest_block, priority_fee]);

    let mut nodes = HashMap::new();
    let rpc_hash = vec![1];
    let priority_hash = vec![2];
    let multiplier_hash = vec![3];
    let gas_hash = vec![4];
    let path_hash = vec![5];
    let max_fee_hash = vec![6];

    nodes.insert(
        rpc_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar_string(rpc_url, 1.0)),
    );
    nodes.insert(
        priority_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar(2.0, 1.0)),
    );
    nodes.insert(
        multiplier_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar(1.5, 1.0)),
    );
    nodes.insert(
        gas_hash.clone(),
        RuntimeNode::Operation {
            op: Op::GetGasPrice,
            inputs: vec![rpc_hash.clone(), priority_hash, multiplier_hash],
        },
    );
    nodes.insert(
        path_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar_string("max_fee_gwei".to_string(), 1.0)),
    );
    nodes.insert(
        max_fee_hash.clone(),
        RuntimeNode::Operation {
            op: Op::JsonGet,
            inputs: vec![gas_hash, path_hash],
        },
    );

    let graph = RuntimeGraph {
        nodes,
        entry_point: rpc_hash,
        outputs: vec![max_fee_hash],
        version: 1,
        proofs: vec![],
    };

    let mut vm = VM::new();
    let outputs = vm.execute(&graph).unwrap();
    assert!((outputs[0].as_scalar() - 32.0).abs() < 0.001);

    server.join().unwrap();
}

#[test]
fn test_oracle_read_can_feed_json_get() {
    let oracle_value = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": "0x000000000000000000000000000000000000000000000000000000003ade68b1"
    });
    let (rpc_url, server) = spawn_mock_rpc_server(vec![oracle_value]);

    let mut nodes = HashMap::new();
    let rpc_hash = vec![11];
    let address_hash = vec![12];
    let calldata_hash = vec![13];
    let decimals_hash = vec![14];
    let oracle_hash = vec![15];
    let path_hash = vec![16];
    let normalized_hash = vec![17];

    nodes.insert(
        rpc_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar_string(rpc_url, 1.0)),
    );
    nodes.insert(
        address_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar_string(
            "0x0000000000000000000000000000000000000001".to_string(),
            1.0,
        )),
    );
    nodes.insert(
        calldata_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar_string("0x50d25bcd".to_string(), 1.0)),
    );
    nodes.insert(
        decimals_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar(8.0, 1.0)),
    );
    nodes.insert(
        oracle_hash.clone(),
        RuntimeNode::Operation {
            op: Op::OracleRead,
            inputs: vec![rpc_hash.clone(), address_hash, calldata_hash, decimals_hash],
        },
    );
    nodes.insert(
        path_hash.clone(),
        RuntimeNode::Constant(Tensor::scalar_string("normalized_f32".to_string(), 1.0)),
    );
    nodes.insert(
        normalized_hash.clone(),
        RuntimeNode::Operation {
            op: Op::JsonGet,
            inputs: vec![oracle_hash, path_hash],
        },
    );

    let graph = RuntimeGraph {
        nodes,
        entry_point: rpc_hash,
        outputs: vec![normalized_hash],
        version: 1,
        proofs: vec![],
    };

    let mut vm = VM::new();
    let outputs = vm.execute(&graph).unwrap();
    assert!((outputs[0].as_scalar() - 9.876_543).abs() < 0.000_1);

    server.join().unwrap();
}
