from __future__ import annotations as _annotations

from pydantic_gepa import Candidate, Plan, Stage
from pydantic_gepa.observers import logfire_observer
from pydantic_gepa.orchestration import StageOutput


def main() -> None:
    plan = Plan(
        Stage(
            "prompt",
            components=("prompt",),
            run=lambda candidate, _: StageOutput(candidate=candidate, score=1.0, metric_calls=1),
        ),
        initial_candidate=Candidate(values={"prompt": "Be precise."}),
    )
    plan.run(on_event=logfire_observer())


if __name__ == "__main__":
    main()
