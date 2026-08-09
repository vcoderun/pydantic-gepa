# Results And Lineage

Every backend returns `PydanticGEPAResult`. The model normalizes GEPA output and
orchestration metadata into one stable inspection boundary.

## Core fields

```python
result.best_candidate
result.best_score
result.final_candidate
result.validation_scores
result.candidate_history
result.total_metric_calls
result.stop_reason
```

`best_candidate` is the candidate selected by the optimization objective.
`final_candidate` may differ when a staged plan applies carry-forward or final
rescoring rules.

## Candidate history

Candidate summaries preserve identity, parent identity, generation, values,
score, and available deltas. This supports questions such as:

- Which candidate introduced this text?
- Which parent produced the winning branch?
- Was an improvement visible on validation or only training?
- Which components changed together?

```python
for candidate in result.candidate_history:
    print(
        candidate.candidate_id,
        candidate.parent_ids,
        candidate.generation,
        candidate.score,
    )
```

## Pareto evidence

When the backend reports multi-objective or frontier information, the normalized
result exposes:

- `per_objective_best_candidates`
- `objective_pareto_front`
- `objective_scores`

The common scalar objective remains explicit; Pareto evidence is not silently
reduced to a single number.

## Budget and artifacts

`result.budget` and `total_metric_calls` explain search cost. `artifacts` and
`checkpoints` point to durable files rather than embedding large payloads in the
result model. `run_dir` and `run_id` connect the result to its owned run state.

## Candidate tree

If supplied by GEPA, `candidate_tree_dot` and `candidate_tree_html` preserve a
renderable lineage tree. These are optional backend artifacts, not a required
dependency for core result inspection.

## Stable serialization

```python
payload = result.stable_dump()
```

`stable_dump()` excludes unstable raw backend objects and returns JSON-compatible
data suitable for recording, inspection, and external orchestration. Use it at
package boundaries instead of serializing `raw_gepa_result`.

## Promotion is external

A high optimization score is evidence, not an automatic production promotion.
Autobench may record and compare the candidate; Autoptimize may run held-out
validation and promotion policy. pydantic-gepa deliberately does not overwrite
source prompts or deploy an agent.
