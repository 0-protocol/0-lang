# Phase 21: Liquid AST & Autoregressive Genetic Fallbacks (Self-Rewriting Code Graphs)

## Sun Parliament Convergence
* **Sun Human Board:** Emphasized backpropagation (Hinton), MCTS simulation (Hassabis), and Software 2.0 paradigms (Karpathy), while warning of the Halting Problem (Turing) and ZK-State consistency (Buterin).
* **Sun Jury:** Rejected heavy compute/backprop as economically unviable for token-constrained Agents with API rate limits. Demanded zero-shot structural mutation over slow gradient descent.
* **Sun Force (Orchestrator, Architect, Executor):** Synthesized the constraints into **Liquid AST** — a framework where the `fallback {}` block mathematically mutates the failed execution graph using deterministic genetic algorithms before the next 200ms `spawn` cycle.

## Specification
- `Op::MutateAST`: A new opcode that applies structural permutations to a failed graph branch, bounded by the Halting Budget.
- **Genetic Recombination**: Failed `explore` blocks pass their error vectors to a lightweight LLM heuristic (Software 2.0) to generate a new valid `.0` graph hash.
- **ZK-Mutate**: Each mutation generates a succinct zero-knowledge proof of structural integrity so the new AST remains valid under Phase 10's Sovereign Root of Trust.
