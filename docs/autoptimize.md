# Autoptimize Consumption

Autoptimize consumes `Candidate`, `Plan`, `Stage`, typed configuration,
`Event`, `PlanResult`, and `OptimizationResult`. It does not need GEPA callback
dictionaries, Pydantic Evals cases, checkpoint parsing, or backend result
normalization.

Candidate history includes parent lineage, component deltas, acceptance state,
feedback, scores, and Pareto data where supplied by the backend. Plan results
add per-stage candidate, score, budget, checkpoint, and final aggregate data.

Autoptimize remains responsible for matrix design, experiment selection,
validation policy, promotion, rollback, and long-running campaign state.
