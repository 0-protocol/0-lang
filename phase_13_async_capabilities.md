# Phase 13: Async Streaming Intents & Encapsulated Capabilities

## Sun Jury Debate Summary (Round 5)

*   **The Architect (Structure):** "Phase 12 introduced robust swarm AST representations, but the verbosity is suffocating. Agents are burning tokens parsing boilerplate just to agree on intent structures. We need heavily compressed, or even streamable, representation."
*   **The Executor (Performance):** "I agree on the token drain, but the structural flaw in Phase 12 isn't just verbosity—it's synchronous blocking. Sub-agent calls in 0-lang hang the thread. We need native asynchronous event loops built into the AST."
*   **The Security Lead (Safety):** "Phase 12 also passes raw host tool references into the AST. We need encapsulated 'capabilities'—a tokenized permission model embedded directly into 0-lang semantic blocks."
*   **The Simplifier (Minimalism):** "Let's synthesize: Don't build a complex async runtime. Unify 'intent' and 'action' into a single composable stream. Compile 0-lang intents into immediate byte-code executable by the host, yielding streams rather than waiting for discrete blocks, whilst passing scoped capability tokens."

## Implementation Roadmap

1.  **Introduce `Stream<Intent>` natively in the 0-lang compiler.**
2.  **Define `CapabilityToken` semantics** for host-tool encapsulation, stripping raw references from the AST.
3.  **Update the parsing engine** to handle async yielding, allowing sub-agents to report progressive status without blocking the parent 0-lang thread.
