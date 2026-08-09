# Development

## Environment

```bash
uv sync --extra dev --extra integrations --extra docs
```

## Quality gate

```bash
make prod
make pre-commit
```

The project requires formatting, Ruff, basedpyright, ty, pytest, and 100% line
and branch coverage. Tests must exercise behavior rather than merely execute
branches.

## Documentation

```bash
make docs
make docs-serve
```

`make docs` regenerates `llms-full.txt`, stages exact Markdown sources for the
Copy as Markdown action, and runs a strict Zensical build. Every public Markdown
page must appear exactly once in navigation.

## Repository layout

```text
src/pydantic_gepa/     package source
tests/                 behavioral and contract tests
examples/              runnable integrations
docs/                  user documentation
scripts/llms.py        documentation corpus generator
.agents/skills/        agent-facing package skill
```

## Contribution rules

- Prefer exact types, then generics, then `Any` only for truly dynamic boundaries.
- Do not use `object` as a type-checking escape hatch.
- Keep Pydantic Evals internal for common API changes.
- Add typed configuration instead of forwarding arbitrary GEPA kwargs.
- Preserve scoped candidate isolation under concurrency.
- Put unstable upstream APIs under `experimental`.
- Update runnable examples and docs with every public behavior change.
- Run the complete quality gate before committing.

## Release checks

Build wheel and sdist, inspect their contents, verify optional dependency
boundaries, run examples that do not require credentials, build docs strictly,
and verify the hosted `llms.txt` and `llms-full.txt` endpoints.
