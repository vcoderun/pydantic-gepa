# Autoptimize Contract

Autoptimize may use pydantic-gepa as one optimization backend. Pydantic-gepa
does not own experiment planning or production promotion.

## Inputs from Autoptimize

- selected tracked assets and allowed component space
- training, validation, and protected test evidence
- objective and constraint definitions
- budgets and concurrency policy
- initial candidate and candidate lineage
- requested backend and typed configuration

## Outputs from pydantic-gepa

- normalized best and final candidates
- complete candidate history and parent links
- objective and validation scores
- budget use and stop reason
- Pareto and per-objective evidence
- checkpoints, artifacts, and stable result serialization

## Promotion remains external

Autoptimize should independently:

1. rescore the candidate on held-out evidence;
2. compare constraints and regressions to the baseline;
3. run isolating or interaction experiments when assets changed together;
4. generate a promotion recommendation;
5. preserve rollback information;
6. never overwrite application source solely because GEPA selected a candidate.

## Backend neutrality

Pydantic-gepa supplies standard GEPA and experimental Optimize Anything
backends. Autoptimize may add other strategies without changing the candidate,
evaluation, and result contracts.
