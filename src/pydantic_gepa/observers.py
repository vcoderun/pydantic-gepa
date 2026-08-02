from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from .errors import OptimizationDependencyError
from .events import Event, Observer, event_payload
from .values import JsonValue

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress, TaskID


class AutobenchRecorder(Protocol):
    def record_optimization_event(self, event: Mapping[str, JsonValue]) -> None: ...


class RichProgress:
    def __init__(
        self,
        *,
        console: Console | None = None,
        interactive: bool | None = None,
    ) -> None:
        try:
            from rich.console import Console as RichConsole
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
        except ImportError as exc:
            raise OptimizationDependencyError(
                "Rich progress requires the 'progress' extra."
            ) from exc

        self.console = console or RichConsole()
        self.interactive = self.console.is_terminal if interactive is None else interactive
        self.progress: Progress | None = None
        self.tasks: dict[str, TaskID] = {}
        if self.interactive:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                console=self.console,
                transient=False,
            )

    def __call__(self, event: Event) -> None:
        if self.progress is None:
            if event.kind in {
                "stage.completed",
                "stage.failed",
                "run.completed",
                "run.failed",
                "checkpoint.resumed",
            }:
                context = event.stage_id or event.candidate_id or event.run_id
                self.console.print(
                    f"[pydantic-gepa] {event.kind}: {context}",
                    markup=False,
                )
            return
        if event.kind == "run.started":
            self.progress.start()
        elif event.kind == "stage.started" and event.stage_id is not None:
            self.tasks[event.stage_id] = self.progress.add_task(event.stage_id, total=1)
        elif event.kind in {"stage.completed", "stage.failed"} and event.stage_id is not None:
            task_id = self.tasks.get(event.stage_id)
            if task_id is not None:
                self.progress.update(task_id, completed=1)
        elif event.kind in {"run.completed", "run.failed"}:
            self.progress.stop()


class LogfireObserver:
    def __init__(self, *, logfire: Any | None = None) -> None:
        if logfire is None:
            try:
                import logfire as installed_logfire
            except ImportError as exc:
                raise OptimizationDependencyError(
                    "Logfire events require the 'logfire' extra."
                ) from exc
            logfire = installed_logfire
        self.logfire = logfire

    def __call__(self, event: Event) -> None:
        self.logfire.info(
            "pydantic-gepa {event_kind}",
            event_kind=event.kind,
            optimization_event=dict(event_payload(event)),
        )


def rich_progress(
    *,
    console: Console | None = None,
    interactive: bool | None = None,
) -> Observer:
    return RichProgress(console=console, interactive=interactive)


def logfire_observer(*, logfire: Any | None = None) -> Observer:
    return LogfireObserver(logfire=logfire)


def autobench_observer(recorder: AutobenchRecorder) -> Observer:
    def record(event: Event) -> None:
        recorder.record_optimization_event(event_payload(event))

    return record


def callback_observer(callback: Callable[[Mapping[str, JsonValue]], None]) -> Observer:
    def record(event: Event) -> None:
        callback(event_payload(event))

    return record


__all__ = (
    "AutobenchRecorder",
    "LogfireObserver",
    "RichProgress",
    "autobench_observer",
    "callback_observer",
    "logfire_observer",
    "rich_progress",
)
