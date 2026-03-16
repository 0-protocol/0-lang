# Phase 12: Swarm-native AST Compilation

## Overview
Introduces primitives for concurrent agent sub-graph execution directly within the 0-lang syntax.

## New Syntax
- `||` operator for parallel agent invocation.
- `spawn_ast(agent_id, task)` for dynamic sub-agent spawning during AST resolution.

## Implementation Stub
```python
def compile_swarm_ast(node):
    if node.type == 'parallel_spawn':
        return [spawn_agent(child) for child in node.children]
    return compile_standard(node)
```