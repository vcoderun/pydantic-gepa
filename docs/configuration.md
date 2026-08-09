# Typed GEPA Configuration

`GEPAConfig` is the supported configuration surface. Its nested models reject
unknown and conflicting values before expensive model calls begin.

```python
from pydantic_gepa import GEPAConfig
from pydantic_gepa.configuration import (
    BudgetConfig,
    EvaluationSetConfig,
    MergeConfig,
    ProgressConfig,
    ReflectionConfig,
    RunConfig,
    SelectionConfig,
    TrackingConfig,
)

config = GEPAConfig(
    reflection=ReflectionConfig(
        model="openai:gpt-5-mini",
        minibatch_size=4,
        skip_perfect_score=True,
    ),
    selection=SelectionConfig(
        candidate="pareto",
        frontier="hybrid",
        component="round_robin",
    ),
    merge=MergeConfig(enabled=True, max_invocations=3),
    budget=BudgetConfig(max_metric_calls=100),
    run=RunConfig(id="support-v1", directory="runs/support-v1"),
    tracking=TrackingConfig(track_best_outputs=True),
    progress=ProgressConfig(display_bar=True),
    evaluation_sets=EvaluationSetConfig(allow_same_train_validation=False),
)
```

## ReflectionConfig

Controls the reflection model, provider kwargs, minibatch size, perfect-score
handling, prompt template, and custom proposer. The model may be an identifier,
a `CallableReflectionModel`, or an integration-specific reflection adapter.

## SelectionConfig

Controls candidate, frontier, component, batch-sampler, validation, and
acceptance strategies. Use typed values instead of passing backend-specific
strings through arbitrary `**kwargs`.

## MergeConfig

Controls whether GEPA may merge candidate branches, how many merge invocations
are allowed, and the validation-overlap floor.

## BudgetConfig

Limits metric calls and optional reflection cost, and configures stop behavior.
Metric calls are the portable budget unit across model providers.

## RunConfig

Owns durable execution:

- run identity and directory
- `resume` and `fresh` behavior
- checkpoint interval
- compatibility validation
- deterministic seed
- evaluation cache policy
- exception behavior

See [Checkpoint and resume](state.md).

## TrackingConfig

Connects loggers, backend callbacks, typed observers, optional best-output
tracking, and supported external tracking integrations. Observer failures may be
configured independently from optimization failures.

## ProgressConfig

`display_bar=True` enables backend progress when supported. The Rich observer
adds package-level stage and event progress.

## EvaluationSetConfig

The default rejects identical training and validation sets. Enabling overlap is
an explicit compatibility choice, not a recommended evaluation design.

## Callable reflection

```python
from pydantic_gepa import CallableReflectionModel

reflection = CallableReflectionModel(
    lambda prompt: "A more precise candidate",
    retries=2,
)
```

The adapter records requests, supplied or estimated usage, cost, retries,
duration, and normalized failures.

## Legacy options

`GEPAConfig.from_legacy_kwargs()` maps known historical options and warns. It
rejects unknown keys instead of forwarding an untyped bag. New code should
construct typed models directly.
