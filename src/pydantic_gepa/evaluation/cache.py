from __future__ import annotations as _annotations

from threading import RLock
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .models import CaseResult

OutputT = TypeVar("OutputT")


@runtime_checkable
class CacheStore(Protocol[OutputT]):
    def get(self, key: str) -> CaseResult[OutputT] | None: ...

    def set(self, key: str, result: CaseResult[OutputT]) -> None: ...


class InMemoryCache(Generic[OutputT]):
    def __init__(self) -> None:
        self._results: dict[str, CaseResult[OutputT]] = {}
        self._lock = RLock()

    def get(self, key: str) -> CaseResult[OutputT] | None:
        with self._lock:
            return self._results.get(key)

    def set(self, key: str, result: CaseResult[OutputT]) -> None:
        with self._lock:
            self._results[key] = result

    def clear(self) -> None:
        with self._lock:
            self._results.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)


__all__ = (
    "CacheStore",
    "InMemoryCache",
)
