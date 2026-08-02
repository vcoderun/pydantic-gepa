# Plans And Stages

A `Stage` targets one or many named components. Multi-component stages provide
grouped optimization without a separate component-group abstraction.

`Plan` executes stages in deterministic order and supports:

- accepted or initial candidate carry-forward;
- frozen non-target components;
- stage and shared metric-call budgets;
- stage-specific runners and rescoring;
- mean, weighted mean, minimum, or custom aggregation;
- stop-on-failure or continue behavior;
- serializable snapshots with explicit callable registries on restore.

Autoptimize may construct a plan, but `Plan` itself does not choose future
experiments or promote candidates.
