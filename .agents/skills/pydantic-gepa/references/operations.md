# Operations And Repository

CLI targets use `module:attribute` and resolve an `Optimization`, `Plan`, or zero-argument factory.
Inspect before run; use owned run directories for resume/fresh commands.

Repository map:

```text
src/pydantic_gepa/   source and public contracts
tests/               behavioral tests and compatibility contracts
examples/            runnable usage and integration programs
docs/                Zensical user documentation
scripts/llms.py      llms-full and Copy as Markdown source generation
```

For repository changes run:

```bash
uv sync --extra dev --extra integrations --extra docs
make prod
make pre-commit
```

Coverage is 100% line and branch. Prefer exact types, then generics, then `Any` only at truly
dynamic boundaries; never use `object` as a type-checking escape hatch. Update source, exports,
docs, real examples, tests, and durable compatibility together.

Canonical docs: `docs/cli.md`, `docs/troubleshooting.md`, `docs/development.md`,
`docs/api-reference.md`.
