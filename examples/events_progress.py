from __future__ import annotations as _annotations

from pydantic_gepa import Candidate, Plan, Stage
from pydantic_gepa.events import Event
from pydantic_gepa.observers import rich_progress
from pydantic_gepa.orchestration import StageOutput


def main() -> None:
    events: list[Event] = []
    plan = Plan(
        Stage(
            "prompt",
            components=("prompt",),
            run=lambda candidate, _: StageOutput(candidate=candidate, score=1.0, metric_calls=1),
        ),
        initial_candidate=Candidate(values={"prompt": "Be precise."}),
    )
    result = plan.run(on_event=[events.append, rich_progress()])
    print([event.kind for event in events])
    print(result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
