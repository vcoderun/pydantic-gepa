from __future__ import annotations as _annotations

import json
import os
import shutil
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ..candidates import Candidate
from ..configuration import ResumeMode
from ..errors import RunStoreError
from .models import CompatibilityFingerprint, RunState

ResultT = TypeVar("ResultT", bound=BaseModel)
_OWNER = "pydantic-gepa"
_MARKER = ".pydantic-gepa-run"
_OWNED_FILES = (_MARKER, "manifest.json", "state.json", "result.json")
_OWNED_DIRECTORIES = ("backend", "candidates", "stages")


class FileRunStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        resume: ResumeMode = "never",
        fresh: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.run_id = run_id
        self.resume = resume
        self.fresh = fresh

    @property
    def backend_directory(self) -> Path:
        return self.directory / "backend"

    def prepare(
        self,
        *,
        fingerprint: CompatibilityFingerprint,
        initial_candidate: Candidate,
    ) -> RunState:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._claim_or_validate()
        self._cleanup_temporary_files()
        was_reset = False
        if self.fresh:
            self.reset()
            self._write_marker()
            was_reset = True

        state_path = self.directory / "state.json"
        if state_path.exists():
            if self.resume == "never":
                raise RunStoreError(
                    f"Run state already exists at '{self.directory}'. Use resume or fresh mode."
                )
            state = self._load_model(state_path, RunState)
            if state.fingerprint.digest != fingerprint.digest:
                changed = sorted(
                    key
                    for key in set(state.fingerprint.dimensions) | set(fingerprint.dimensions)
                    if state.fingerprint.dimensions.get(key) != fingerprint.dimensions.get(key)
                )
                raise RunStoreError(
                    "Run state is incompatible with the current optimization: " + ", ".join(changed)
                )
            return state.model_copy(update={"resumed": True})
        if self.resume == "required":
            raise RunStoreError(f"No resumable state exists at '{self.directory}'.")

        state = RunState(
            run_id=self.run_id,
            fingerprint=fingerprint,
            accepted_candidate=initial_candidate,
            reset=was_reset,
        )
        self._write_manifest(state)
        self.checkpoint(state)
        return state

    def checkpoint(self, state: RunState) -> None:
        if state.run_id != self.run_id:
            raise RunStoreError("Cannot write state owned by a different run id.")
        self._require_owned()
        self._write_model(self.directory / "state.json", state)
        self._write_manifest(state)

    def write_candidate(self, candidate: Candidate) -> Path:
        self._require_owned()
        destination = self.directory / "candidates" / f"{candidate.fingerprint()}.json"
        self._write_model(destination, candidate)
        return destination

    def write_stage(self, result: BaseModel) -> Path:
        self._require_owned()
        stage_id = result.model_dump().get("stage_id")
        if not isinstance(stage_id, str) or not stage_id:
            raise RunStoreError("Stage snapshots require a non-empty stage_id field.")
        stage_name = stage_id.replace("/", "_").replace("\\", "_")
        destination = self.directory / "stages" / f"{stage_name}.json"
        self._write_model(destination, result)
        return destination

    def write_result(self, result: BaseModel) -> Path:
        self._require_owned()
        destination = self.directory / "result.json"
        self._write_model(destination, result)
        return destination

    def load_result(self, model: type[ResultT]) -> ResultT | None:
        path = self.directory / "result.json"
        if not path.exists():
            return None
        self._require_owned()
        return self._load_model(path, model)

    def reset(self) -> None:
        self._require_owned()
        for name in _OWNED_FILES:
            path = self.directory / name
            if path.exists():
                path.unlink()
        for name in _OWNED_DIRECTORIES:
            path = self.directory / name
            if path.exists():
                shutil.rmtree(path)

    def _claim_or_validate(self) -> None:
        marker = self.directory / _MARKER
        if marker.exists():
            self._require_owned()
            return
        if any(self.directory.iterdir()):
            raise RunStoreError(
                f"Refusing to own non-empty directory '{self.directory}' without {_MARKER}."
            )
        self._write_marker()

    def _cleanup_temporary_files(self) -> None:
        for name in (*_OWNED_FILES, "candidate.json", "stage.json"):
            temporary = self.directory / f".{name}.tmp"
            if temporary.exists():
                temporary.unlink()

    def _require_owned(self) -> None:
        marker = self.directory / _MARKER
        if not marker.is_file():
            raise RunStoreError(f"Run directory '{self.directory}' is not owned by pydantic-gepa.")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunStoreError(f"Run ownership marker at '{marker}' is corrupted.") from exc
        if payload != {"owner": _OWNER, "run_id": self.run_id}:
            raise RunStoreError(f"Run ownership marker at '{marker}' does not match this run.")

    def _write_marker(self) -> None:
        self._atomic_write(
            self.directory / _MARKER,
            json.dumps({"owner": _OWNER, "run_id": self.run_id}, sort_keys=True),
        )

    def _write_manifest(self, state: RunState) -> None:
        payload = {
            "schema_version": 1,
            "owner": _OWNER,
            "run_id": state.run_id,
            "fingerprint": state.fingerprint.model_dump(mode="json"),
            "status": state.status,
            "next_stage": state.next_stage,
        }
        self._atomic_write(
            self.directory / "manifest.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        )

    @staticmethod
    def _write_model(path: Path, model: BaseModel) -> None:
        FileRunStore._atomic_write(path, model.model_dump_json(indent=2))

    @staticmethod
    def _load_model(path: Path, model: type[ResultT]) -> ResultT:
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RunStoreError(f"Stored run file '{path}' is corrupted.") from exc

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise RunStoreError(f"Could not atomically write '{path}'.") from exc


__all__ = ("FileRunStore",)
