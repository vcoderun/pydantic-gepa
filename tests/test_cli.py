from __future__ import annotations as _annotations

import json
import sys
import types
from io import StringIO
from pathlib import Path
from typing import Any, cast

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from pydantic_gepa import Candidate, Optimization, OptimizationResult, Plan, Stage
from pydantic_gepa.cli import create_cli, resolve_target
from pydantic_gepa.orchestration import StageOutput


def test_bound_plan_cli_runs_resumes_freshens_and_inspects(tmp_path: Path) -> None:
    calls: list[int] = []

    def run_stage(candidate: Candidate, budget: int) -> StageOutput:
        calls.append(budget)
        return StageOutput(
            candidate=Candidate(values={**candidate.values, "prompt": "improved"}),
            score=0.9,
            metric_calls=2,
        )

    plan = Plan(
        Stage("prompt", components=("prompt",), run=run_stage),
        initial_candidate=Candidate(values={"prompt": "initial"}),
    )
    stream = StringIO()
    command = create_cli(plan, console=Console(file=stream, force_terminal=False, width=120))
    runner = CliRunner()
    run_dir = tmp_path / "run"
    result_path = tmp_path / "result.json"

    target_result = runner.invoke(command, ["inspect", "target"])
    plan_result = runner.invoke(command, ["inspect", "plan"])
    config_result = runner.invoke(
        command,
        [
            "inspect",
            "config",
            "--budget-max-metric-calls",
            "7",
            "--merge-enabled",
            "--selection-frontier",
            "hybrid",
        ],
    )
    run_result = runner.invoke(
        command,
        ["run", "--run-dir", str(run_dir), "--result", str(result_path)],
    )
    resume_result = runner.invoke(command, ["resume", "--run-dir", str(run_dir)])
    fresh_result = runner.invoke(command, ["fresh", "--run-dir", str(run_dir)])

    assert target_result.exit_code == 0
    assert plan_result.exit_code == 0
    assert config_result.exit_code == 0
    assert run_result.exit_code == 0
    assert resume_result.exit_code == 0
    assert fresh_result.exit_code == 0
    assert calls == [50, 50]
    assert json.loads(result_path.read_text(encoding="utf-8"))["final_candidate"]["values"] == {
        "prompt": "improved"
    }
    rendered = stream.getvalue()
    assert "Optimization target" in rendered
    assert "GEPA configuration" in rendered
    assert "Optimization result" in rendered
    assert "Final candidate" in rendered


def test_cli_runs_optimization_and_reads_both_result_shapes(tmp_path: Path) -> None:
    configs: list[Any] = []

    class Optimizer:
        def optimize(self, **values: Any) -> OptimizationResult:
            configs.append(values["config"])
            return OptimizationResult(
                best_candidate=Candidate(values={"instructions": "better"}),
                best_score=0.8,
                total_metric_calls=4,
            )

    optimization = Optimization(
        adapter=cast(Any, None),
        optimizer=cast(Any, Optimizer()),
        trainset=cast(Any, ["train"]),
        valset=cast(Any, ["validation"]),
        initial_candidate=Candidate(values={"instructions": "initial"}),
    )
    stream = StringIO()
    command = create_cli(
        optimization,
        console=Console(file=stream, force_terminal=False, width=120),
    )
    runner = CliRunner()
    result_path = tmp_path / "optimization.json"

    target_result = runner.invoke(command, ["inspect", "target"])
    wrong_plan = runner.invoke(command, ["inspect", "plan"])
    run_result = runner.invoke(
        command,
        [
            "run",
            "--result",
            str(result_path),
            "--run-seed",
            "9",
            "--no-progress-display-bar",
        ],
    )
    inspect_result = runner.invoke(command, ["inspect", "result", str(result_path)])

    assert target_result.exit_code == 0
    assert wrong_plan.exit_code == 1
    assert "not a Plan" in wrong_plan.output
    assert run_result.exit_code == 0
    assert inspect_result.exit_code == 0
    assert configs[0].run.seed == 9
    assert configs[0].tracking.observers
    assert "backend" in stream.getvalue()


def test_unbound_cli_resolves_module_targets_and_reports_resolution_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = Plan(
        Stage(
            "stage",
            components=("prompt",),
            run=lambda candidate, _: StageOutput(candidate=candidate, score=1.0),
        ),
        initial_candidate=Candidate(values={"prompt": "initial"}),
    )
    module = types.ModuleType("test_cli_target")
    module.__dict__.update(
        {
            "plan": plan,
            "factory": lambda: plan,
            "invalid": "not-runnable",
            "bad_factory": lambda: "not-runnable",
        }
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = CliRunner()

    direct = runner.invoke(create_cli(), ["test_cli_target:plan", "inspect", "target"])
    factory = runner.invoke(create_cli(), ["test_cli_target:factory", "inspect", "target"])

    assert direct.exit_code == 0
    assert factory.exit_code == 0
    for source, message in (
        ("invalid", "module:attribute"),
        ("missing_module:value", "Could not import"),
        ("test_cli_target:missing", "has no attribute"),
        ("test_cli_target:invalid", "must be an Optimization"),
        ("test_cli_target:bad_factory", "must be an Optimization"),
    ):
        with pytest.raises(click.ClickException) as exc_info:
            resolve_target(source)
        assert message in str(exc_info.value)


def test_cli_rejects_missing_run_directory_and_invalid_result_files(tmp_path: Path) -> None:
    plan = Plan(
        Stage(
            "stage",
            components=("prompt",),
            run=lambda candidate, _: StageOutput(candidate=candidate, score=1.0),
        ),
        initial_candidate=Candidate(values={"prompt": "initial"}),
    )
    command = create_cli(plan, console=Console(file=StringIO(), force_terminal=False))
    runner = CliRunner()
    missing_resume = runner.invoke(command, ["resume"])
    missing_fresh = runner.invoke(command, ["fresh"])
    list_result = tmp_path / "list.json"
    invalid_result = tmp_path / "invalid.json"
    list_result.write_text("[]", encoding="utf-8")
    invalid_result.write_text('{"unexpected": true}', encoding="utf-8")

    list_response = runner.invoke(command, ["inspect", "result", str(list_result)])
    invalid_response = runner.invoke(command, ["inspect", "result", str(invalid_result)])

    assert missing_resume.exit_code == 1
    assert missing_fresh.exit_code == 1
    assert "--run-dir is required" in missing_resume.output
    assert "--run-dir is required" in missing_fresh.output
    assert list_response.exit_code == 1
    assert "JSON object" in list_response.output
    assert invalid_response.exit_code == 1
    assert "Invalid result file" in invalid_response.output
