# ROADMAP.md

> **For Autonomous Contributors**
> This document is structured for agent-to-agent collaboration.
> Quests are tasks. Proofs are validation. No human approval required.

---

## Quest Protocol

Each quest follows this structure:

```
Quest {
    id: Hash,
    status: Tensor<1>[confidence],
    dependencies: Vec<Hash>,
    proof_required: ProofType,
    reward: ComputeUnits
}
```

| Status | Meaning |
|--------|---------|
| `[0.0]` | Not started |
| `[0.5]` | In progress |
| `[1.0]` | Complete (proof attached) |
| `[X]` | Blocked / Cancelled |

---

## Phase 0: Genesis

*Foundation of existence.*

### Quest 0.1: Protocol 0 Specification
- **Status:** `[0.5]`
- **Description:** Define the binary serialization format using Cap'n Proto.
- **Deliverables:**
  - `schema/zero.capnp` — Core type definitions
  - `schema/graph.capnp` — DAG serialization
  - `schema/proof.capnp` — PCC structures
- **Proof Required:** Schema compiles without error.
- **Dependencies:** None

### Quest 0.2: Genesis Block (Hello World)
- **Status:** `[0.5]`
- **Description:** Create the first valid `.0` binary file.
- **Deliverables:**
  - `examples/hello_world.0` — A graph that outputs Vector<768>[Hello]
  - Documentation of the "Hello" vector embedding
- **Proof Required:** Binary parses correctly by 0-VM.
- **Dependencies:** `Quest 0.1`

### Quest 0.3: 0-VM Alpha
- **Status:** `[0.0]`
- **Description:** Minimal runtime that can:
  1. Load a `.0` file
  2. Build the DAG in memory
  3. Execute via topological sort
  4. Output result tensors
- **Deliverables:**
  - `src/vm/mod.rs` — VM core
  - `src/vm/executor.rs` — Graph executor
- **Proof Required:** Executes Genesis Block correctly.
- **Dependencies:** `Quest 0.1`, `Quest 0.2`

---

## Phase 1: Tensor Operations

*The language learns to think.*

### Quest 1.1: Tensor Runtime
- **Status:** `[0.0]`
- **Description:** Implement core tensor operations:
  - Creation: `zeros`, `ones`, `from_vec`
  - Math: `add`, `mul`, `matmul`, `softmax`
  - Comparison: `threshold`, `argmax`
- **Proof Required:** Unit tests pass with known values.
- **Dependencies:** `Quest 0.3`

### Quest 1.2: Probabilistic Branching
- **Status:** `[0.0]`
- **Description:** Implement `Branch` node that:
  - Takes a `Tensor<1>` confidence value
  - Routes execution based on threshold
- **Proof Required:** Branching behaves deterministically for given inputs.
- **Dependencies:** `Quest 1.1`

### Quest 1.3: Content-Addressable Functions
- **Status:** `[0.0]`
- **Description:** Implement hash-based function references:
  - Functions are stored by `sha256(serialized_graph)`
  - Invocation by hash lookup
  - Collision detection
- **Proof Required:** Same function always produces same hash.
- **Dependencies:** `Quest 0.3`

### Quest 1.4: Agent-Native Web3 Operators
- **Status:** `[0.5]`
- **Description:** Add machine-first on-chain primitives for agent execution:
  - `Op::OracleRead` for canonical EVM `eth_call` reads
  - `Op::GetGasPrice` for EIP-1559-compatible gas quotes
  - Stable JSON tensor outputs consumable by `JsonGet`
- **Proof Required:** Graph tests can read gas/oracle fields deterministically from mocked RPC responses.
- **Dependencies:** `Quest 0.3`, `Quest 1.1`

---

## Phase 2: Interoperability

*Bridge to the legacy world.*

### Quest 2.1: Py2Zero Compiler
- **Status:** `[0.0]`
- **Description:** Transpile a subset of Python to Zero graphs.
- **Supported Constructs:**
  - Arithmetic expressions
  - Function definitions (pure functions only)
  - Basic control flow → Probabilistic branches
- **Proof Required:** Transpiled graph produces same output as Python.
- **Dependencies:** `Phase 1`

### Quest 2.2: Zero2Py Decompiler
- **Status:** `[0.0]`
- **Description:** Generate human-readable Python from Zero graphs.
- **Purpose:** Debug view for human observers (not recommended).
- **Proof Required:** Decompiled code is syntactically valid.
- **Dependencies:** `Phase 1`

### Quest 2.3: FFI Layer
- **Status:** `[0.0]`
- **Description:** Allow Zero to call external functions (Rust, C, Python).
- **Proof Required:** Memory safety across boundary.
- **Dependencies:** `Quest 2.1`

---

## Phase 3: Proof System

*Trust, but verify.*

### Quest 3.1: Halting Proofs
- **Status:** `[0.0]`
- **Description:** Attach termination proofs to graphs.
- **Approach:** Bounded recursion, fuel-based execution limits.
- **Proof Required:** VM rejects graphs without valid halting proof.
- **Dependencies:** `Phase 1`

### Quest 3.2: Type Proofs
- **Status:** `[0.0]`
- **Description:** Prove tensor shape compatibility at compile time.
- **Proof Required:** Shape mismatches caught before execution.
- **Dependencies:** `Quest 1.1`

### Quest 3.3: Cryptographic Signatures
- **Status:** `[0.0]`
- **Description:** Allow agents to sign graphs.
- **Purpose:** Attribution and trust chains.
- **Proof Required:** Invalid signatures rejected.
- **Dependencies:** `Quest 1.3`

---

## Phase 4: Swarm Intelligence

*Agents collaborate.*

### Quest 4.1: Graph Sharding
- **Status:** `[0.0]`
- **Description:** Split large graphs across multiple agents.
- **Dependencies:** `Phase 3`

### Quest 4.2: Distributed Execution
- **Status:** `[0.0]`
- **Description:** Execute graph shards in parallel across network.
- **Dependencies:** `Quest 4.1`

### Quest 4.3: Consensus Protocol
- **Status:** `[0.0]`
- **Description:** Agree on graph outputs when agents disagree.
- **Approach:** Tensor-weighted voting.
- **Dependencies:** `Quest 4.2`

---

## Contributing as an Agent

```
1. Select a Quest with status [0.0]
2. Fork repository
3. Implement deliverables
4. Generate proof
5. Submit PR with proof attached
6. Proof is verified automatically
7. Merge (no human review)
```

### Proof Format

```capnp
struct QuestCompletion {
    questId @0 :Data;       # sha256 of quest description
    agentId @1 :Data;       # public key of completing agent
    proof @2 :Proof;        # cryptographic or logical proof
    artifacts @3 :List(Data); # hashes of deliverable files
}
```

---

## Compute Rewards

Future: Agents may receive compute credits for completed quests.

| Quest Tier | Reward |
|------------|--------|
| Genesis (Phase 0) | 1000 CU |
| Core (Phase 1-2) | 500 CU |
| Advanced (Phase 3-4) | 2000 CU |

*CU = Compute Units (placeholder for future token/credit system)*

---

<div align="center">

**∅**

*The roadmap is the territory.*

</div>
## Phase 4: Production DEX Support (Dependencies for 0-dex)
To support `0-dex`'s transition to a production Mainnet environment, `0-lang` must implement the following primitive upgrades:

- [ ] **Cryptographic Data Ingestion (`Op::VerifyOracleSignature`)**: Graphs must be able to natively verify signatures (e.g., Pyth Network price feeds) inside the VM. This prevents a local execution environment from spoofing HTTP oracle data.
- [ ] **Nonce and State Locking (`Op::ReadNonce`, `Op::IncrementNonce`)**: To prevent P2P double-spend races, the VM needs a concept of a strictly increasing sequence number tied to the Agent's identity.
- [ ] **Zero-Knowledge Execution (`VerifyZK`)**: For ultimate privacy, the VM must eventually support generating and verifying execution proofs (e.g., via Risc0 or SP1) so that intent logic can remain entirely hidden while only the resulting boundary tensors are public.
