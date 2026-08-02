# Architecture

The runtime has five distinct lifecycles:

1. `Runtime` applies a candidate, invokes the subject, enforces isolation, and
   restores application state.
2. `Evaluation` owns output-scoring or evaluator-controlled execution and
   produces a `CaseResult`.
3. The GEPA backend searches candidate component values.
4. `Plan` deterministically sequences one or more optimization stages.
5. `RunStore` persists compatibility fingerprints, accepted candidates, stage
   results, checkpoints, and final results.

Typed `Event` values observe these lifecycles but never become authoritative
state. This separation lets Rich, Logfire, and Autobench consume the same event
stream without weakening resume guarantees.

## Dependency direction

Application code depends on the common API. Integration modules may depend on
Pydantic AI or Pydantic Evals. Standard and experimental backends consume the
same candidate, evaluation, event, state, and result contracts. Autobench and
Autoptimize are optional consumers; pydantic-gepa does not import them.
