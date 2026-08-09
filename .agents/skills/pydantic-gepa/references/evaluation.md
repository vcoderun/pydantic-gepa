# Evaluation

`Evaluation.output` runs once then scores. `Evaluation.controlled` gives the evaluator a `Context`
that may call `run`/`arun` zero or multiple times. Use controlled evaluation for robustness,
pairwise judging, and repeated sampling.

Return a float for one objective or `MetricResult`/a mapping for richer evidence. Keep objective,
constraint, and diagnostic roles distinct. Put concise mutation guidance in `feedback` and
serializable detail in `side_info`.

`EvaluationConfig` owns task/evaluator/infrastructure failure policy, invalid scores, bounds,
cache behavior, and validation hooks. NaN, infinity, absent objective keys, and out-of-range scores
must not reach GEPA silently.

Candidate scope uses context variables, but application clients must themselves support chosen
concurrency. Mark nondeterministic evaluation and avoid caching it by default.

Canonical docs: `docs/evaluation.md`, `docs/candidates-concurrency.md`, `docs/data.md`.
