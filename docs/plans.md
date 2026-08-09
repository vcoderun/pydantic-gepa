# Plans And Stages

`Plan` composes ordered optimization stages. Use it when one monolithic GEPA run
would mix component responsibilities, budgets, or validation policies.

```python
from pydantic_gepa import Budget, Candidate, Plan, Stage
from pydantic_gepa.orchestration import StageOutput

def optimize_planner(candidate: Candidate, budget: int) -> StageOutput:
    return StageOutput(
        candidate=run_planner_optimizer(candidate, budget),
        score=0.82,
        metric_calls=budget,
    )

plan = Plan(
    Stage(
        "planner",
        components=("planner.instructions",),
        run=optimize_planner,
        budget=Budget(max_metric_calls=20),
        run_id="planner-v1",
    ),
    Stage(
        "tools",
        components=("tool:search", "tool:lookup"),
        run=optimize_tools,
        budget=Budget(max_metric_calls=30),
        run_id="tools-v1",
    ),
    initial_candidate=baseline,
    budget=Budget(max_metric_calls=50),
)
```

## Stage contract

A stage declares:

- stable stage id and implementation id
- components it may modify
- frozen components it must preserve
- stage budget and objective
- optional seed candidate
- optional rescoring function and stable id

Its callable receives the current candidate and remaining metric-call budget,
then returns `StageOutput` with a candidate, score, calls used, acceptance,
history, and optional checkpoint.

## Carry forward

The plan decides whether accepted or final stage candidates carry into the next
stage. Frozen component checks prevent a stage from silently changing values it
does not own.

## Global budget

Per-stage budgets are capped by the plan's remaining global budget. A stage
must report actual metric calls. The plan records stop reasons when budget or
failure policy prevents later stages.

## Aggregation and final rescore

Plan scores may use mean or weighted aggregation. A final rescore can validate
the assembled candidate after all stages, which is important because components
that work separately may interact when combined.

## When not to use a plan

Use a single `Optimization` when all components share one evaluator, budget,
and search strategy. Plans add value only when stage boundaries reflect actual
ownership or validation boundaries.
