from __future__ import annotations as _annotations

from pydantic_gepa import Budget, Candidate, Plan, Stage
from pydantic_gepa.orchestration import StageOutput


def planner(candidate: Candidate, budget: int) -> StageOutput:
    return StageOutput(
        candidate=Candidate(values={**candidate.values, "planner": "Plan with constraints."}),
        score=0.8,
        metric_calls=min(2, budget),
    )


def generation(candidate: Candidate, budget: int) -> StageOutput:
    assert candidate.values["planner"] == "Plan with constraints."
    return StageOutput(
        candidate=Candidate(
            values={
                **candidate.values,
                "writer": "Write a concise answer.",
                "tool.search": "Search only verified sources.",
            }
        ),
        score=0.9,
        metric_calls=min(3, budget),
    )


pipeline = Plan(
    Stage("planner", components=("planner",), run=planner, budget=Budget(max_metric_calls=2)),
    Stage(
        "generation",
        components=("writer", "tool.search"),
        run=generation,
        budget=Budget(max_metric_calls=3),
    ),
    initial_candidate=Candidate(
        values={"planner": "Draft.", "writer": "Answer.", "tool.search": "Search."}
    ),
    budget=Budget(max_metric_calls=5),
)


if __name__ == "__main__":
    print(pipeline.run().model_dump(mode="json"))
