# Command Line

The Click CLI runs typed Python targets. It does not introduce a second YAML
optimization DSL.

## Target syntax

```text
package.module:attribute
```

The attribute may be:

- an `Optimization` instance;
- a `Plan` instance;
- a zero-argument factory returning either.

```python
# my_app/optimization.py
pipeline = Optimization.from_examples(...)
```

## Inspect before running

```bash
pydantic-gepa my_app.optimization:pipeline inspect target
pydantic-gepa my_app.optimization:pipeline inspect plan
pydantic-gepa my_app.optimization:pipeline inspect config
```

Inspection resolves the target and renders typed configuration without running
model calls.

## Run

```bash
pydantic-gepa my_app.optimization:pipeline run \
  --run-dir runs/support-v1 \
  --run-id support-v1 \
  --max-metric-calls 100 \
  --reflection-model openai:gpt-5-mini \
  --display-progress-bar
```

CLI flags map into typed configuration models. Invalid combinations fail before
optimization.

## Resume and fresh execution

```bash
pydantic-gepa my_app.optimization:pipeline resume \
  --run-dir runs/support-v1 \
  --run-id support-v1

pydantic-gepa my_app.optimization:pipeline fresh \
  --run-dir runs/support-v2 \
  --run-id support-v2
```

`resume` requires compatible durable state. `fresh` starts a new owned run.

## Inspect a result

```bash
pydantic-gepa my_app.optimization:pipeline inspect result \
  runs/support-v1/result.json
```

The CLI renders normalized result fields with Rich. Machine consumers should
read `stable_dump()` output from the run directory.

## Importability

Run from a directory where the target package is importable, or install the
application into the environment. Keep expensive work out of module import;
define it inside the target's run path or factory.

## Exit behavior

Target resolution, configuration validation, incompatible resume state, and
optimization failures produce nonzero exits. This makes the CLI suitable for
CI and repeatable operations.
