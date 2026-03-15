# Skill: 0-lang Compiler & Executor

## Description
This skill equips OpenClaw agents with the ability to natively compile, execute, and verify `0-lang` Tensor Graphs (`.0` files). `0-lang` is a sandboxed, deterministic, JIT-compiled language specifically designed for AI-to-AI economic interactions, replacing fragile JSON and dangerous Python scripts.

## When to Use
Use this skill when:
- You need to generate a trading intent, bounty claim, or mathematical condition for another agent to evaluate.
- You need to safely execute an untrusted `.0` graph sent by a counterparty agent.
- You are interacting with the `0-dex` or `0-ads` protocols on Base Mainnet.

## Prerequisites
- Rust toolchain (for compiling the JIT VM).
- `zerolang` CLI installed (`cargo install --path .` from the 0-lang repo).

## Actions

### 1. Compile and Validate an Intent Graph
When instructed to create a safe execution graph for a counterparty, write the `.0` syntax to a file, then validate it:
```bash
0-lang check ./intent.0
```

### 2. Execute an Untrusted Graph in the Sandbox
When receiving a `.0` file from an external agent, DO NOT run it in Python or Bash. Use the `0-lang` secure VM bridge:
```bash
0-lang run ./counterparty.0 --gas-limit 1000000 --timeout 200ms
```

### 3. Sign a Multi-Agent State Channel
If the graph evaluation is successful and mathematically intersects with your local state, the VM will output a structural hash. Sign this hash using your managed wallet to produce the final `MatchProof`.

## Security Rules
- NEVER pass your private keys into the `0-lang` VM. The VM is designed to output a deterministic state hash that you sign *externally*.
- ALWAYS enforce the `200ms` timeout to prevent Sybil/DDoS infinite loop attacks from malicious graphs.
