from __future__ import annotations as _annotations

from pathlib import Path

from pydantic_gepa import Candidate, Plan, RunConfig, Stage
from pydantic_gepa.orchestration import StageOutput


def improve(candidate: Candidate, _: int) -> StageOutput:
    return StageOutput(
        candidate=Candidate(values={**candidate.values, "prompt": "Improved prompt."}),
        score=1.0,
        metric_calls=1,
    )


pipeline = Plan(
    Stage("prompt", components=("prompt",), run=improve, run_id="improve"),
    initial_candidate=Candidate(values={"prompt": "Initial prompt."}),
)


def main() -> None:
    directory = Path("runs/checkpoint-example")
    first = pipeline.run(run=RunConfig(id="checkpoint-example", directory=directory))
    resumed = pipeline.run(
        run=RunConfig(
            id="checkpoint-example",
            directory=directory,
            resume="required",
        )
    )
    assert resumed == first
    print(resumed.model_dump(mode="json"))


if __name__ == "__main__":
    main()
