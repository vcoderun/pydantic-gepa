# Installation

## Common API

Install the package and runtime integrations with `uv`:

```bash
uv add pydantic-gepa
uv add "pydantic-gepa[integrations]"
```

Or with `pip`:

```bash
pip install "pydantic-gepa[integrations]"
```

The base package contains candidate models, configuration, orchestration,
result models, and the CLI. The `integrations` extra installs GEPA, Pydantic AI,
and Pydantic Evals for actual optimization runs.

## Optional features

```bash
uv add "pydantic-gepa[logfire]"       # Logfire observer integration
uv add "pydantic-gepa[progress]"      # Rich progress rendering
uv add "pydantic-gepa[optimize-anything]"  # GEPA Omni and built-in agent engines
uv add "pydantic-gepa[examples]"      # All example dependencies
```

`integrations` installs base GEPA, Pydantic AI, and Pydantic Evals. The
`optimize-anything` extra additionally installs `gepa[full]`. AutoResearch and
other external agent engines may still require their own CLI, credentials, and
operating-system sandbox support.

## Python support

Python 3.11, 3.12, and 3.13 are supported. Verify the installation:

```bash
uv run python -c "import pydantic_gepa; print(pydantic_gepa.__version__)"
pydantic-gepa --help
```

## Provider credentials

Model credentials are consumed by Pydantic AI or the reflection provider, not
by a separate pydantic-gepa credential system. Configure the environment
variables expected by your provider before a live run.

Do not commit keys in optimization targets, candidate YAML, run directories,
or examples. Candidate and result files are intended to be inspectable and may
be persisted.

## Development checkout

```bash
git clone https://github.com/vcoderun/pydantic-gepa.git
cd pydantic-gepa
uv sync --extra dev --extra integrations --extra docs
make prod
```

`make prod` runs formatting checks, linting, both type checkers, tests, branch
coverage, and documentation validation.
