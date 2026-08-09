# Architecture

The package separates application-facing concepts from backend plumbing.

```text
examples + task + score
          |
          v
 Optimization / optimize
          |
          v
 components -> candidate -> injections -> runtime -> evaluation
                                                |
                                                v
                                    Pydantic Evals harness
                                                |
                                                v
                            standard GEPA / Optimize Anything
                                                |
                                                v
                                  PydanticGEPAResult
```

## Layers

### Common API

`examples.py` exposes `Example`, `Optimization`, `optimize`, score context, and
built-in scorers. It keeps Pydantic Evals internal for ordinary users.

### Candidate model

`candidates.py`, `components.py`, and `injections.py` define mutable search
dimensions, concrete versions, selection, serialization, and scoped application
binding.

### Evaluation runtime

`evaluation/` and `runtime/` own examples, controlled execution, caches,
evidence encoding, traces, failure policy, and final rescoring.

### GEPA adapter

`adapter.py`, `asi.py`, `harness.py`, `reflection.py`, and `optimizer.py`
translate normalized evaluation evidence into GEPA contracts and translate
backend output back into stable result models.

### Orchestration and state

`orchestration/` composes stages and budgets. `state/` owns run manifests,
checkpoints, compatibility, and atomic durable files.

### Optional integrations

`integrations/` contains SDK-specific adapters. Experimental upstream surfaces
remain under `experimental/`.

## Dependency direction

The candidate and evaluation core do not depend on Autobench or Autoptimize.
Integrations depend inward on stable contracts. External systems connect through
recorders, observers, typed results, and stable dumps.

## Design rules

- Pydantic Evals is internal in the common API and public only as an advanced seam.
- Configuration is typed; unknown backend kwargs are rejected.
- Candidate application is scoped and concurrency-aware.
- Results normalize backend objects before crossing package boundaries.
- Experimental backends do not redefine standard public models.
- Durable resume validates compatibility instead of trusting a directory name.
