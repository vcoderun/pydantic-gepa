# Integrations

For Pydantic AI, optimize agent instructions with `AgentInstructionsInjection`; optimize structured
outputs with `ModelOutputInjection`; use `PydanticAIReflectionModel` when reflection itself should
run through Pydantic AI. SDK imports remain optional.

Multimodal examples use typed SDK inputs or `Attachment`. Keep large binary content in references
or artifact stores and include media identity in stable example identity.

Optimize Anything Omni is experimental. Select `backend="optimize_anything"`, then pass
`OptimizeAnythingConfig(engine=Engine...)` or a typed composition. GEPA, AutoResearch,
Meta-Harness, Best-of-N, and custom engines share the same internal evaluation runtime. Use
`Pipeline(BestOf(...), Single(...))` for Omni exploration and continuation. Persist only
normalized common results; never upstream engine or result objects.

Autobench integration subscribes to the versioned typed event stream and uses normalized
result/component seams to store immutable semantic evidence. The common lifecycle covers standard
GEPA, Optimize Anything engines and compositions, staged plans, evaluation cases, selections,
resource-specific budgets, checkpoints, resume, and terminal outcomes. Direct Pydantic AI/model
and HTTP evidence remains owned by the corresponding native Autobench instrumentors.
Autoptimize owns held-out validation, matrix planning, promotion, and rollback. Pydantic-gepa does
not overwrite source prompts or deploy candidates.

Canonical docs: `docs/pydantic-ai.md`, `docs/multimodal.md`, `docs/optimize-anything.md`,
`docs/autobench.md`, `docs/autoptimize.md`.
