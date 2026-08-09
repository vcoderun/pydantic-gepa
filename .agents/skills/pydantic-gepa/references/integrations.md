# Integrations

For Pydantic AI, optimize agent instructions with `AgentInstructionsInjection`; optimize structured
outputs with `ModelOutputInjection`; use `PydanticAIReflectionModel` when reflection itself should
run through Pydantic AI. SDK imports remain optional.

Multimodal examples use typed SDK inputs or `Attachment`. Keep large binary content in references
or artifact stores and include media identity in stable example identity.

Optimize Anything is experimental. Select `backend="optimize_anything"`, provide objective and
background, keep `GEPAConfig`, and persist only normalized common results.

Autobench integration uses recorder/event/result seams to store immutable semantic evidence.
Autoptimize owns held-out validation, matrix planning, promotion, and rollback. Pydantic-gepa does
not overwrite source prompts or deploy candidates.

Canonical docs: `docs/pydantic-ai.md`, `docs/multimodal.md`, `docs/optimize-anything.md`,
`docs/autobench.md`, `docs/autoptimize.md`.
