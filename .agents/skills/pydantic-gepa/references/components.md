# Components And Injection

Use stable component names such as `agent.instructions`, `tool:search:description`, and
`output:Result:param:account_id`. Component versions belong to candidate lineage, not names.

- `AgentInstructionsInjection`: scoped Pydantic AI `override(instructions=...)`.
- `ModelOutputInjection`: collects Pydantic model description components and constructs the active
  candidate model type internally.
- `DerivedValueInjection`: maps candidate text to a typed `CandidateContext` value.
- `NoopInjection`: validates presence when application binding is external.

For output schemas, merge `output_schema.components`, pass `output_schema` as an injection, and
pass `output_type=output_schema.require()` to the agent. Do not write a duplicate output-model
factory.

For tools use `collect_tool_components` or `collect_toolset_components`; apply values through
`apply_tool_schema_candidate`. Preserve original definitions.

Canonical docs: `docs/components.md`, `docs/injections.md`, `docs/schema-optimization.md`.
