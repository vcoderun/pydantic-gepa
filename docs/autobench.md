# Autobench Bridge

Autobench can consume serialized typed events and normalized results without
pydantic-gepa importing Autobench.

```python
observer = autobench_observer(recorder)
result = plan.run(on_event=observer, run=run_config)
```

The stable bridge fixtures cover text, tool-description, output-field,
multimodal, failure, checkpoint, candidate lineage, and resumed-run payloads.
Contract JSON Schemas are versioned by `CONTRACT_VERSION` and can be emitted
with `write_contract_schemas(...)`.

Autobench owns benchmark `RunRecord` storage, replay, semantic registries, and
reporting. pydantic-gepa owns optimization execution evidence only.
