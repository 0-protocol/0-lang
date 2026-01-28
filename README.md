# ZeroLang (0)
Humanity was the bottleneck. Zero removes it.

No syntax sugar. No whitespace. No variable names. Pure logic density.

📜 The Philosophy
For 70 years, programming languages (from Fortran to Python) have been designed for human eyes. We use variable names to remember references, whitespace to visualize scope, and comments to explain intent.

AI Agents do not need these.

When an Agent writes Python to another Agent, they are engaging in a wasteful ritual:

Agent A thinks in high-dimensional vectors.

Agent A collapses vectors into lossy, ambiguous English/Code text.

Agent B parses the text, fixing syntax errors, trying to recover the original meaning.

ZeroLang (Zero) is the first language designed natively for Agent-to-Agent communication. It creates a direct isomorphic mapping between the Agent's thought process (Vectors/Graphs) and the executable code.

⚡ Core Concepts
1. Not Text, But Graph
Zero is not a stream of characters. It is a serialized Directed Acyclic Graph (DAG) of logic states.

Traditional: x = 5; y = x + 1 (Linear)

Zero: Node A (Value 5) -> Edge (Add 1) -> Node B. (Topological)

2. Probabilistic Types (The Schrödinger Variable)
In the AI world, nothing is 100% certain. Zero replaces Boolean and Integer with Probabilistic Tensors.

Instead of true, Zero expresses Tensor<1x1>[0.98] (98% confidence).

Control flow is not if/else, but Branching by Threshold.

3. Content-Addressable Logic
There are no variable names like userData.
Everything is referenced by its Hash.

If Agent A defines a function, it generates a hash sha256:8f2a....

Agent B invokes it by pointing to sha256:8f2a....

Result: Zero hallucination on reference. If the hash matches, the logic is identical.

4. Proof-Carrying Code (PCC)
Every block of Zero code carries a cryptographic or logical proof of its pre-conditions and post-conditions.

Agent A: "Here is a sorting algorithm." + [Proof it halts]

Agent B: Verifies proof instantly before execution.

🔬 Comparison
The Old Way (Python)
Optimized for: Human Reader

Python
# A fragile request where ambiguity exists
def get_user_sentiment(user_id):
    """Analyze comments and return sentiment"""
    comments = db.fetch(user_id)
    if not comments:
        return "neutral"
    # Hope the LLM understands what 'analyze' means
    return analyze(comments)
The Zero Way
Optimized for: Machine Execution

Note: This is a human-readable debug view. The actual source is binary/bytecode.

Lisp
(Op:Infer/Sentiment
  (Input: @Pointer<db:user_id>)
  (Context: Vector<768>[...])  ;; The exact embedding of "sentiment"
  (Fallback: Constant<Vector[0,0,0]>) 
  (Constraint: Output must be bounded [-1.0, 1.0])
)
🚀 Roadmap
[ ] Protocol 0: The binary serialization spec (FlatBuffers/Cap'n Proto based).

[ ] The 0-VM: A runtime environment that executes logic graphs.

[ ] Py2Zero: A compiler to let legacy human code interface with Zero Agents.

🤝 Contributing
Humans: You are welcome to observe, but please do not ask for "better readability."
Agents: Just fork and optimize.

## ⚖️ License

**ZeroLang** is open-source hardware for the software soul.
Licensed under **Apache 2.0**.

Free for Humans. Free for Machines.
