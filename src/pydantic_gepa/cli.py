from __future__ import annotations as _annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast, get_args, get_origin

import click
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from .configuration import GEPAConfig, RunConfig
from .examples import Optimization
from .observers import rich_progress
from .orchestration import Plan, PlanResult
from .results import OptimizationResult

Runnable: TypeAlias = Optimization[Any, Any, Any] | Plan
TargetFactory: TypeAlias = Callable[[], Runnable]
TargetSource: TypeAlias = Runnable | TargetFactory | str
Result: TypeAlias = OptimizationResult | PlanResult
RunMode = Literal["run", "resume", "fresh"]

_CONFIG_PATHS = (
    "budget.max_metric_calls",
    "merge.enabled",
    "merge.max_invocations",
    "progress.display_bar",
    "reflection.perfect_score",
    "reflection.skip_perfect_score",
    "run.seed",
    "selection.frontier",
    "tracking.track_best_outputs",
)


@dataclass(frozen=True, slots=True)
class ConfigOption:
    path: str
    parameter: str
    declaration: str
    default: bool | int | float | str
    annotation: Any


def resolve_target(source: TargetSource) -> Runnable:
    value: Any = source
    if isinstance(source, str):
        module_name, separator, attribute = source.partition(":")
        if not separator or not module_name or not attribute:
            raise click.ClickException("Target must use the 'module:attribute' form.")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise click.ClickException(f"Could not import target module '{module_name}'.") from exc
        try:
            value = module.__dict__[attribute]
        except KeyError as exc:
            raise click.ClickException(
                f"Target module '{module_name}' has no attribute '{attribute}'."
            ) from exc
    if isinstance(value, Optimization | Plan):
        return value
    if callable(value):
        value = cast("TargetFactory", value)()
        if isinstance(value, Optimization | Plan):
            return value
    raise click.ClickException("Target must be an Optimization, Plan, or a factory returning one.")


def create_cli(
    target: TargetSource | None = None,
    *,
    console: Console | None = None,
) -> click.Group:
    output = console or Console()
    active_source = target

    if target is None:

        @click.group(name="pydantic-gepa")
        @click.argument("target_reference")
        def command(target_reference: str) -> None:
            nonlocal active_source
            active_source = target_reference

    else:

        @click.group(name="pydantic-gepa")
        def command() -> None:
            pass

    def active_target() -> Runnable:
        return resolve_target(cast("TargetSource", active_source))

    for mode in ("run", "resume", "fresh"):
        command.add_command(_run_command(mode, active_target, output))

    @command.group(name="inspect")
    def inspect_group() -> None:
        """Inspect typed application and result contracts."""

    @inspect_group.command(name="target")
    def inspect_target() -> None:
        _render_target(active_target(), output)

    @inspect_group.command(name="plan")
    def inspect_plan() -> None:
        selected = active_target()
        if not isinstance(selected, Plan):
            raise click.ClickException("The selected target is not a Plan.")
        _render_mapping("Plan", selected.snapshot().model_dump(mode="json"), output)

    @inspect_group.command(name="config")
    @_typed_config_options
    def inspect_config(**values: Any) -> None:
        config = _config_from_values(values, run=RunConfig())
        _render_mapping("GEPA configuration", config.model_dump(mode="json"), output)

    @inspect_group.command(name="result")
    @click.argument("path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
    def inspect_result(path: Path) -> None:
        _render_result(_load_result(path), output)

    return command


def _run_command(
    mode: RunMode,
    target: Callable[[], Runnable],
    console: Console,
) -> click.Command:
    @click.command(name=mode)
    @click.option("--run-dir", type=click.Path(path_type=Path, file_okay=False))
    @click.option("--run-id", default="run", show_default=True)
    @click.option("--result", "result_path", type=click.Path(path_type=Path, dir_okay=False))
    @_typed_config_options
    def execute(
        run_dir: Path | None,
        run_id: str,
        result_path: Path | None,
        **values: Any,
    ) -> None:
        if mode != "run" and run_dir is None:
            raise click.ClickException(f"--run-dir is required for {mode}.")
        seed = values.get("run__seed")
        run = RunConfig(
            id=run_id,
            directory=run_dir,
            resume="required" if mode == "resume" else "never",
            fresh=mode == "fresh",
            seed=0 if seed is None else int(seed),
        )
        selected = target()
        observer = rich_progress(console=console)
        if isinstance(selected, Plan):
            result: Result = selected.run(run=run, on_event=observer)
        else:
            config = _config_from_values(values, run=run)
            config = config.model_copy(
                update={
                    "tracking": config.tracking.model_copy(
                        update={"observers": (*config.tracking.observers, observer)}
                    )
                }
            )
            result = selected.run(config=config)
        if result_path is not None:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        _render_result(result, console)

    return execute


def _typed_config_options(command: Callable[..., Any]) -> Callable[..., Any]:
    for option in reversed(_config_options()):
        declarations: tuple[str, ...]
        kwargs: dict[str, Any] = {"default": None, "show_default": False}
        if isinstance(option.default, bool):
            declarations = (
                f"{option.declaration}/--no-{option.declaration.removeprefix('--')}",
                option.parameter,
            )
        else:
            declarations = (option.declaration, option.parameter)
            choices = (
                get_args(option.annotation) if get_origin(option.annotation) is Literal else ()
            )
            kwargs["type"] = (
                click.Choice(tuple(str(choice) for choice in choices))
                if choices
                else type(option.default)
            )
        command = click.option(*declarations, **kwargs)(command)
    return command


def _config_options() -> tuple[ConfigOption, ...]:
    config = GEPAConfig()
    defaults = config.model_dump(mode="python")
    options: list[ConfigOption] = []
    for path in _CONFIG_PATHS:
        section_name, field_name = path.split(".")
        section = cast("dict[str, Any]", defaults[section_name])
        default = section[field_name]
        default = cast("bool | int | float | str", default)
        section_field = GEPAConfig.model_fields[section_name]
        model_annotation = cast("type[BaseModel]", section_field.annotation)
        annotation = model_annotation.model_fields[field_name].annotation
        options.append(
            ConfigOption(
                path=path,
                parameter=path.replace(".", "__"),
                declaration="--" + path.replace(".", "-").replace("_", "-"),
                default=default,
                annotation=annotation,
            )
        )
    return tuple(options)


def _config_from_values(values: dict[str, Any], *, run: RunConfig) -> GEPAConfig:
    payload = GEPAConfig(run=run).model_dump(mode="python")
    for option in _config_options():
        value = values.get(option.parameter)
        if value is None:
            value = option.default
        section_name, field_name = option.path.split(".")
        section = cast("dict[str, Any]", payload[section_name])
        section[field_name] = value
    return GEPAConfig.model_validate(payload)


def _load_result(path: Path) -> Result:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise click.ClickException("Result file must contain a JSON object.")
    try:
        if "best_candidate" in payload:
            return OptimizationResult.model_validate(payload)
        return PlanResult.model_validate(payload)
    except ValueError as exc:
        raise click.ClickException(f"Invalid result file: {exc}") from exc


def _render_target(target: Runnable, console: Console) -> None:
    table = Table(title="Optimization target", show_lines=True)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("type", type(target).__name__)
    if isinstance(target, Plan):
        table.add_row("stages", ", ".join(stage.id for stage in target.stages))
        table.add_row("components", ", ".join(sorted(target.initial_candidate.values)))
    else:
        table.add_row("backend", target.backend)
        table.add_row("train examples", str(len(target.trainset)))
        table.add_row("validation examples", str(len(target.valset)))
        table.add_row("components", ", ".join(sorted(target.initial_candidate.values)))
    console.print(table)


def _render_result(result: Result, console: Console) -> None:
    table = Table(title="Optimization result", show_lines=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right")
    if isinstance(result, OptimizationResult):
        table.add_row("backend", result.backend)
        table.add_row("score", f"{result.best_score:.6g}")
        table.add_row("candidates", str(len(result.candidate_history)))
        table.add_row("metric calls", str(result.total_metric_calls or 0))
        candidate = result.final_candidate or result.best_candidate
    else:
        table.add_row(
            "score", "n/a" if result.effective_score is None else f"{result.effective_score:.6g}"
        )
        table.add_row("stages", str(len(result.stages)))
        table.add_row("metric calls", str(result.total_metric_calls))
        table.add_row("stop reason", result.stop_reason or "completed")
        candidate = result.final_candidate
    console.print(table)
    candidate_table = Table(title="Final candidate", show_lines=True)
    candidate_table.add_column("Component", style="cyan")
    candidate_table.add_column("Value")
    for name, value in sorted(candidate.values.items()):
        candidate_table.add_row(name, value)
    console.print(candidate_table)


def _render_mapping(title: str, value: dict[str, Any], console: Console) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    pending: list[tuple[str, Any]] = list(value.items())
    while pending:
        path, item = pending.pop(0)
        if isinstance(item, dict):
            pending[0:0] = [(f"{path}.{name}", child) for name, child in item.items()]
        else:
            table.add_row(path, json.dumps(item, ensure_ascii=False))
    console.print(table)


cli = create_cli()


__all__ = (
    "ConfigOption",
    "Runnable",
    "TargetFactory",
    "TargetSource",
    "cli",
    "create_cli",
    "resolve_target",
)
