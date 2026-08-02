from __future__ import annotations as _annotations

from pydantic_gepa import Candidate, Context, Evaluation, Example, MetricResult, Runtime


def main() -> None:
    runtime = Runtime[str, str](
        lambda value: value.strip().lower(),
        identity="normalize-text",
    )
    candidate = Candidate(values={"instructions": "Normalize text."})
    example = Example(inputs="  HELLO  ", expected_output="hello", id="hello")

    output_scoring = Evaluation.output(
        runtime,
        lambda ctx: {
            "accuracy": float(ctx.output == ctx.example.expected_output),
            "length": MetricResult(score=float(len(ctx.output or "")), role="diagnostic"),
        },
        objective="accuracy",
    )

    def stability(ctx: Context[str, str, None]) -> MetricResult:
        first = ctx.run()
        second = ctx.run()
        return MetricResult(
            score=float(first == second),
            feedback="Compared two evaluator-owned invocations.",
            side_info={"first": first, "second": second},
        )

    controlled = Evaluation.controlled(runtime, stability)
    print(output_scoring.run(candidate, example).model_dump(mode="json"))
    print(controlled.run(candidate, example).model_dump(mode="json"))


if __name__ == "__main__":
    main()
