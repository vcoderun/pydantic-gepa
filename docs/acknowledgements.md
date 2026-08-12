# Acknowledgements And Design Influences

`pydantic-gepa` is an independent open-source project built on, integrated
with, and informed by work from several open-source communities. This page
distinguishes runtime dependencies from design references so those influences
remain visible.

## Core Upstream Projects

- [GEPA](https://github.com/gepa-ai/gepa) provides the upstream optimization
  algorithm and standard Python backend adapted by this package. The
  experimental Optimize Anything integration also targets GEPA's upstream
  experimental API.
- [Pydantic AI and Pydantic Evals](https://github.com/pydantic/pydantic-ai)
  provide the typed agent and evaluation runtimes integrated by this package.

`pydantic-gepa` does not claim authorship of those projects, their algorithms,
or their APIs.

## Design And Research References

The following projects have been studied for integration patterns, evaluation
feedback design, optimization workflows, or developer-experience comparisons:

- [dmontagu/pydantic-ai-gepa-example](https://github.com/dmontagu/pydantic-ai-gepa-example)
  demonstrated a compact Pydantic AI, Pydantic Evals, and GEPA integration and
  was an important early reference for candidate injection and reflective
  evaluation.
- [indexedlabs/pydantic-ai-gepa](https://github.com/indexedlabs/pydantic-ai-gepa)
  has informed comparative research into high-level Pydantic AI ergonomics,
  component discovery, candidate comparison, and proposal provenance.
- [dspydantic](https://github.com/davidberenstein1957/dspydantic) was reviewed
  for typed structured-input and optimization integration patterns.
- [DeepEval](https://github.com/confident-ai/deepeval) was reviewed for agent
  evaluation, prompt optimization, tracing, and product-level integration
  patterns.

Being listed here does not imply affiliation, endorsement, API compatibility,
or shared maintainership. `pydantic-gepa` does not claim ownership of these
projects' names, source code, documentation, or project-specific designs.

## Licensing

Each referenced or integrated project remains governed by its own license and
copyright notices. The `pydantic-gepa` license applies only to this project's
own source and documentation; it does not relicense third-party work.

When third-party code is incorporated rather than merely studied, its
applicable license and required notices must be preserved. Please report a
missing attribution or notice through the
[project issue tracker](https://github.com/vcoderun/pydantic-gepa/issues).
