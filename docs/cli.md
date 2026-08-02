# CLI

The Click CLI resolves a typed Python target; it does not introduce a second
optimization DSL.

```bash
pydantic-gepa my_app.optimization:pipeline inspect target
pydantic-gepa my_app.optimization:pipeline run --run-dir runs/demo
pydantic-gepa my_app.optimization:pipeline resume --run-dir runs/demo
pydantic-gepa my_app.optimization:pipeline fresh --run-dir runs/demo
```

A target is an `Optimization`, a `Plan`, or a zero-argument factory returning
one. Backend flags are generated from typed `GEPAConfig` fields. `inspect
config`, `inspect plan`, and `inspect result PATH` render Rich tables. `--result`
writes the normalized result as JSON for automation.

Applications can embed the same commands with `create_cli(target)`.
