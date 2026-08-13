# Example Gallery

The repository examples are executable programs, not isolated snippets. Run
them from the repository root after installing the `examples` extra:

```bash
uv sync --extra examples
uv run python examples/basic.py
```

## Basic instruction optimization

[`examples/basic.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/basic.py)
shows the shortest common API:

- typed `Example` values
- one instruction `Component`
- `AgentInstructionsInjection`
- a callable scorer
- typed GEPA configuration
- normalized best-candidate output

## Reusable optimization target

[`examples/dot_optimization.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/dot_optimization.py)
is an end-to-end multimodal structured extraction pipeline. It loads image
examples, runs a Pydantic AI agent, optimizes instructions and output-field
descriptions together, uses `model_field_accuracy`, and optionally emits
Logfire traces.

## Evaluation strategies

[`examples/evaluation_strategies.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/evaluation_strategies.py)
compares output scoring with evaluator-controlled repeated execution.

## Tool schema optimization

[`examples/schema_components.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/schema_components.py)
collects tool and parameter descriptions into components and applies a
candidate back to a copied tool definition.

## Pydantic output schema optimization

[`examples/model_schema_components.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/model_schema_components.py)
collects nested Pydantic model descriptions and reconstructs the candidate
JSON schema.

## Staged orchestration

[`examples/staged_grouped.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/staged_grouped.py)
optimizes planner and generation component groups in ordered stages with
per-stage and global budgets.

## Checkpoint and resume

[`examples/checkpoint_resume.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/checkpoint_resume.py)
writes a durable run and proves that a compatible required resume returns the
recorded result.

## Events and progress

[`examples/events_progress.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/events_progress.py)
collects typed events while displaying Rich progress.

## Low-level GEPA adapter

[`examples/low_level_adapter.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/low_level_adapter.py)
shows direct candidate-batch evaluation and reflective-dataset construction.
Use this only when building an optimizer integration.

## Recorder hook

[`examples/recorder_hook.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/recorder_hook.py)
connects candidate-batch evaluation to an external evidence recorder.

## Experimental Optimize Anything

[`examples/experimental_optimize_anything.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/experimental_optimize_anything.py)
uses the isolated Optimize Anything Omni backend through the high-level
`Example`/`DataSplit` API. It runs a real GEPA engine, evaluates protected test
data outside optimization, composes deterministic custom engines as
`BestOf -> Single`, prints branch and continuation lineage, and shows a typed
AutoResearch declaration. It uses a local Pydantic AI function model and needs
no API key.

## CLI targets

Any module-level `Optimization`, `Plan`, or zero-argument factory can be exposed
as `module:attribute` and run through the [Click CLI](cli.md).
