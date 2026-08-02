from __future__ import annotations as _annotations


class PydanticGEPAError(Exception):
    """Base exception for pydantic-gepa failures."""


class CandidateComponentError(PydanticGEPAError):
    """Raised when a candidate component cannot be encoded or decoded."""


class CandidateInjectionError(PydanticGEPAError):
    """Raised when a candidate cannot be injected into a target system."""


class EvaluationHarnessError(PydanticGEPAError):
    """Raised when a Pydantic Evals harness cannot run a batch."""


class EvidenceEncodingError(PydanticGEPAError):
    """Raised when runtime evidence cannot be encoded safely."""


class InfrastructureError(PydanticGEPAError):
    """Raised when evaluation infrastructure fails outside task or evaluator logic."""


class InvalidScoreError(PydanticGEPAError):
    """Raised when an evaluator produces an invalid optimization score."""


class OptimizationDependencyError(PydanticGEPAError):
    """Raised when an optional optimization dependency is unavailable."""


class PlanError(PydanticGEPAError):
    """Raised when an optimization plan or stage violates its contract."""


class RunStoreError(PydanticGEPAError):
    """Raised when durable optimization state is unsafe or incompatible."""


__all__ = (
    "CandidateComponentError",
    "CandidateInjectionError",
    "EvidenceEncodingError",
    "EvaluationHarnessError",
    "InfrastructureError",
    "InvalidScoreError",
    "OptimizationDependencyError",
    "PlanError",
    "PydanticGEPAError",
    "RunStoreError",
)
