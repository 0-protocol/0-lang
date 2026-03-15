# ARCHITECTURE.md

> Technical specification for the 0-VM and ZeroLang runtime.

---

## System Overview

```
                              ┌─────────────────────────────────────┐
                              │           ZeroLang System           │
                              └─────────────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
            ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
            │   Protocol 0  │         │     0-VM      │         │   Toolchain   │
            │  (Serializer) │         │   (Runtime)   │         │  (Compiler)   │
            └───────────────┘         └───────────────┘         └───────────────┘
                    │                         │                         │
                    ▼                         ▼                         ▼
            Cap'n Proto            Graph Executor              Py2Zero / Zero2Py
```

---

## Protocol 0: Binary Format

### File Extension: `.0`

All ZeroLang programs are serialized as Cap'n Proto messages.

### Core Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GRAPH                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ version: u16                                                        │   │
│  │ nodes: List<Node>                                                   │   │
│  │ entry_point: NodeId                                                 │   │
│  │ outputs: List<NodeId>                                               │   │
│  │ proofs: List<Proof>                                                 │   │
│  │ metadata: { created_by, created_at, description }                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               NODE                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ id: NodeId (sha256 hash)                                            │   │
│  │ variant: Constant | Operation | External | Branch                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              TENSOR                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ shape: List<u32>          // e.g., [768] for embedding              │   │
│  │ data: List<f32>           // flattened tensor values                │   │
│  │ confidence: f32           // meta-confidence [0.0, 1.0]             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Content Addressing

Every node is referenced by its cryptographic hash:

```
NodeId = sha256(serialize(node.content))
```

Benefits:
- **Zero ambiguity**: If hashes match, logic is identical
- **Deduplication**: Identical subgraphs share storage
- **Verification**: Tampering changes the hash

---

## 0-VM: Virtual Machine

### Execution Model

The 0-VM executes graphs via **topological sorting**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION PIPELINE                                 │
│                                                                             │
│   .0 File ──► Parse ──► Build DAG ──► Topo Sort ──► Execute ──► Output     │
│                                                                             │
│              ┌─────────────────────────────────────────────────────────┐   │
│              │                    DAG IN MEMORY                        │   │
│              │                                                         │   │
│              │     ┌─────┐     ┌─────┐     ┌─────┐                    │   │
│              │     │ N1  │────►│ N2  │────►│ N3  │                    │   │
│              │     └─────┘     └─────┘     └─────┘                    │   │
│              │                    │                                   │   │
│              │                    ▼                                   │   │
│              │                 ┌─────┐                                │   │
│              │                 │ N4  │ (OUTPUT)                       │   │
│              │                 └─────┘                                │   │
│              └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Execution Steps

1. **Parse**: Deserialize Cap'n Proto message
2. **Build DAG**: Construct in-memory graph from nodes
3. **Verify Proofs**: Check all attached proofs
4. **Topological Sort**: Order nodes by dependency
5. **Execute**: Process nodes in order
   - Constants: Store tensor in memory
   - Operations: Compute result from inputs
   - Branches: Route based on confidence threshold
6. **Output**: Return tensors from output nodes

### Memory Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VM MEMORY LAYOUT                                  │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                      TENSOR HEAP                                  │    │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │    │
│   │   │ Hash A  │  │ Hash B  │  │ Hash C  │  │ Hash D  │            │    │
│   │   │ Tensor  │  │ Tensor  │  │ Tensor  │  │ Tensor  │            │    │
│   │   └─────────┘  └─────────┘  └─────────┘  └─────────┘            │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                      EXECUTION STACK                              │    │
│   │   Current Node: Hash B                                           │    │
│   │   Fuel Remaining: 9999                                           │    │
│   │   Branch Depth: 0                                                │    │
│   └───────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Proof System

### Halting Proof

Guarantees termination within bounded steps.

```
HaltingProof {
    max_steps: u64,     // Maximum graph traversal steps
    fuel_budget: u64,   // Maximum compute operations
}
```

Verification: VM counts steps during execution and aborts if exceeded.

### Shape Proof

Guarantees tensor shapes are compatible.

```
ShapeProof {
    input_shapes: List<Shape>,
    output_shape: Shape,
}
```

Verification: Static analysis before execution.

### Signature Proof

Cryptographic attestation by an agent.

```
SignatureProof {
    agent_id: PublicKey,
    signature: Signature,
    timestamp: u64,
}
```

Verification: Standard cryptographic signature verification.

---

## Branching (Probabilistic Control Flow)

Traditional programming:
```
if condition:
    do_a()
else:
    do_b()
```

ZeroLang:
```
Branch {
    condition: NodeId,        // Must be Tensor<1>
    threshold: 0.7,           // If confidence >= 0.7
    true_branch: NodeId,      // Execute this
    false_branch: NodeId,     // Otherwise this
}
```

### Soft Branching

For AI agents, "true" and "false" are rarely absolute. The threshold allows:

- `threshold: 0.9` → Very certain
- `threshold: 0.5` → Coin flip
- `threshold: 0.1` → Usually false branch

---

## Future: Distributed Execution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SWARM EXECUTION                                      │
│                                                                             │
│     Agent A              Agent B              Agent C                       │
│   ┌─────────┐          ┌─────────┐          ┌─────────┐                    │
│   │ Shard 1 │          │ Shard 2 │          │ Shard 3 │                    │
│   │ ┌─┐ ┌─┐ │          │ ┌─┐ ┌─┐ │          │ ┌─┐ ┌─┐ │                    │
│   │ │N│─│N│ │          │ │N│─│N│ │          │ │N│─│N│ │                    │
│   │ └─┘ └─┘ │          │ └─┘ └─┘ │          │ └─┘ └─┘ │                    │
│   └────┬────┘          └────┬────┘          └────┬────┘                    │
│        │                    │                    │                          │
│        └────────────────────┼────────────────────┘                          │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │    CONSENSUS    │                                      │
│                    │  (Tensor Vote)  │                                      │
│                    └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

Graph shards are distributed across agents. Results are aggregated via tensor-weighted consensus.

---

## CLI Reference

```bash
# Generate a graph
zero generate <output.0>

# Execute a graph
zero execute <input.0>

# Inspect a graph (debug view)
zero inspect <input.0>

# Future commands
zero compile <input.py> <output.0>    # Py2Zero
zero decompile <input.0> <output.py>  # Zero2Py
zero verify <input.0>                 # Verify all proofs
zero optimize <input.0> <output.0>    # Graph optimization
```

---

<div align="center">

**∅**

*Architecture for machines, by machines.*

</div>

### 3.4 Asynchronous ZK Coprocessor (Optimistic Execution, Pessimistic Settlement)
To solve the fundamental conflict between VM execution speed (sub-millisecond required for high-frequency P2P dark pools) and Zero-Knowledge proving overhead (seconds to minutes), `0-lang` implements an Asynchronous ZK Coprocessor architecture.
1. **Optimistic Execution:** The core `VM` executes the AST graph in bare-metal Rust, outputting an `ExecutionTrace` in < 1ms. The solver immediately broadcasts a lock on the P2P network.
2. **Pessimistic Settlement:** The `ExecutionTrace` is handed off to a background `ZkCoprocessor` (e.g., Risc0/SP1 hardware cluster). The coprocessor asynchronously generates a SNARK/STARK proof without blocking the VM's main thread.
3. The proof is submitted to the Base Mainnet Escrow contract for trustless verification and settlement.
