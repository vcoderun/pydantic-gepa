from __future__ import annotations as _annotations

# pyright: reportMissingImports=false
from support_routing_common import (
    BASELINE_INSTRUCTIONS,
    IMPROVED_INSTRUCTIONS,
    build_agent,
    build_cases,
    build_dataset,
    run_agent,
)

from pydantic_gepa import (
    AgentInstructionsInjection,
    CandidateComponent,
    PydanticGEPAAdapter,
    ScoreObjective,
)


def main() -> None:
    agent = build_agent()
    component = CandidateComponent(name="instructions", initial_text=BASELINE_INSTRUCTIONS)
    adapter = PydanticGEPAAdapter.from_dataset(
        dataset=build_dataset(),
        task=lambda ticket: run_agent(agent, ticket),
        injections=[AgentInstructionsInjection(agent=agent, candidate_component=component)],
        objective=ScoreObjective(score_key="accuracy"),
    )

    candidate = {"instructions": component.encode(IMPROVED_INSTRUCTIONS)}
    batch = adapter.evaluate(build_cases(), candidate, capture_traces=True)
    reflective_dataset = adapter.make_reflective_dataset(
        candidate=candidate,
        eval_batch=batch,
        components_to_update=["instructions"],
    )

    print("scores:", batch.scores)
    print("outputs:", batch.outputs)
    print("reflection examples:")
    for record in reflective_dataset["instructions"]:
        print(record)


if __name__ == "__main__":
    main()
