# Mental Model

The common pipeline is:

```text
Example -> Candidate -> Injection -> Runtime -> Evaluation -> GEPA -> Result
```

## Example

An `Example` is one typed optimization sample. It contains application inputs,
an optional expected output, metadata, identity, and binary or referenced
attachments. pydantic-gepa converts examples into Pydantic Evals cases
internally.

## Component and candidate

A `Component` describes one mutable dimension, such as `instructions` or
`tool:search:param:query`. A `Candidate` contains concrete text values for one
or more components, plus lineage metadata.

The component is the schema of the search space. The candidate is one point in
that space.

## Injection

An injection binds candidate values to the application for the duration of an
evaluation. It answers: "Where does this optimized text go?"

- `AgentInstructionsInjection` temporarily overrides Pydantic AI instructions.
- `ModelOutputInjection` builds a candidate-specific Pydantic output type.
- `DerivedValueInjection` derives arbitrary typed runtime state.
- `NoopInjection` validates that a component exists when the task reads values
  by another mechanism.

## Runtime and evaluation

`Runtime` executes one example under active injections. `Evaluation.output`
runs the subject then scores its output. `Evaluation.controlled` lets an
evaluator decide whether to call the subject zero, one, or several times.

The high-level `optimize(...)` API builds this machinery for the common case.

## Reflection

GEPA uses failed and successful evaluation evidence to propose improved text.
The reflection model receives selected examples, outputs, expected outputs,
feedback, failure categories, and component context. `pydantic-gepa` normalizes
that evidence and preserves structured side information.

## Result

`PydanticGEPAResult` is the stable output boundary. It normalizes backend data
into typed candidates, scores, budget summaries, history, lineage, Pareto data,
checkpoints, and artifacts.

## Plans

A `Plan` composes deterministic `Stage` objects. Use it when component groups
must be optimized sequentially, share a global budget, or resume from durable
checkpoints. A plan orchestrates optimizers; it is not a second candidate model.

## Ecosystem boundary

| Package | Owns |
| --- | --- |
| pydantic-gepa | Typed execution of GEPA optimization for Pydantic applications |
| Autobench | Replayable benchmark and experiment evidence |
| Autoptimize | Experiment planning, candidate validation, and promotion policy |

The packages can integrate, but pydantic-gepa remains independently usable.
