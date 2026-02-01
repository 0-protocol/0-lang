@0xb5e7a2f3c8d91e04;

# ZeroLang Protocol 0
# Binary serialization format for Agent-to-Agent communication
# No variable names. No whitespace semantics. Pure logic.

struct Tensor {
  # The fundamental data type in Zero.
  # Replaces int, float, bool, string with probabilistic vectors.
  
  shape @0 :List(UInt32);      # Dimensions, e.g. [768] for embedding, [1] for scalar
  data @1 :List(Float32);      # Flattened tensor data
  confidence @2 :Float32;      # Meta-confidence in this tensor's validity [0.0, 1.0]
}

struct NodeId {
  # Content-addressable reference.
  # No variable names - only hashes.
  
  hash @0 :Data;               # sha256 of the node's content
}

struct Node {
  # The fundamental unit of computation.
  # A single vertex in the logic DAG.
  
  id @0 :NodeId;
  
  union {
    # Constants
    constant @1 :Tensor;
    
    # Operations
    operation :group {
      op @2 :Operation;
      inputs @3 :List(NodeId);  # References to input nodes by hash
    }
    
    # External references (FFI, other graphs)
    external :group {
      uri @4 :Text;             # e.g. "sha256:..." or "ffi:rust:..."
      inputMapping @5 :List(NodeId);
    }
    
    # Branching (probabilistic control flow)
    branch :group {
      condition @6 :NodeId;     # Must resolve to Tensor<1> (scalar confidence)
      threshold @7 :Float32;    # Branch if confidence >= threshold
      trueBranch @8 :NodeId;
      falseBranch @9 :NodeId;
    }
  }
}

enum Operation {
  # Tensor operations
  add @0;
  sub @1;
  mul @2;
  div @3;
  matmul @4;
  
  # Activation functions
  softmax @5;
  relu @6;
  sigmoid @7;
  tanh @8;
  
  # Comparison (outputs Tensor<1> confidence)
  eq @9;
  gt @10;
  lt @11;
  gte @20;         # Greater than or equal
  lte @21;         # Less than or equal
  
  # Reduction
  sum @12;
  mean @13;
  argmax @14;
  min @22;         # Element-wise or reduction minimum
  max @23;         # Element-wise or reduction maximum
  
  # Shape manipulation
  reshape @15;
  transpose @16;
  concat @17;
  
  # Special
  identity @18;    # Pass-through (useful for graph composition)
  embed @19;       # Convert hash reference to embedding vector
  
  # Math operations (added for trading applications)
  abs @24;         # Absolute value
  neg @25;         # Negation
  clamp @26;       # Clamp to range (useful for position limits)
}

struct Proof {
  # Every Zero graph carries proof of its properties.
  # Agents verify before execution.
  
  union {
    # The graph terminates within N steps
    halting :group {
      maxSteps @0 :UInt64;
      fuelBudget @1 :UInt64;
    }
    
    # Type/shape proof
    shapeValid :group {
      inputShapes @2 :List(List(UInt32));
      outputShape @3 :List(UInt32);
    }
    
    # Cryptographic signature
    signature :group {
      agentId @4 :Data;         # Public key of signing agent
      sig @5 :Data;             # Signature over graph hash
      timestamp @6 :UInt64;     # Unix timestamp
    }
    
    # No proof (unsafe, may be rejected by strict VMs)
    none @7 :Void;
  }
}

struct Graph {
  # A complete Zero program.
  # This is what gets serialized to a .0 file.
  
  version @0 :UInt16;          # Protocol version (currently 0)
  
  nodes @1 :List(Node);        # All nodes in the graph
  
  entryPoint @2 :NodeId;       # The "main" node - execution starts here
  outputs @3 :List(NodeId);    # Nodes whose values are the "return" values
  
  proofs @4 :List(Proof);      # Attached proofs for this graph
  
  metadata :group {
    # Optional metadata for debugging (not used in execution)
    createdBy @5 :Data;        # Agent ID that created this graph
    createdAt @6 :UInt64;      # Unix timestamp
    description @7 :Data;      # Binary description (could be another embedding)
  }
}
