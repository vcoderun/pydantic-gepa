# Checkpoint, Resume, And Fresh Runs

`RunConfig` controls durable execution:

```python
RunConfig(directory="runs/customer", id="prompt-v1", resume="if_exists")
```

`FileRunStore` writes only beneath an owned directory marked by
`.pydantic-gepa-run`. JSON writes are atomic. Fresh mode removes owned state but
preserves unrelated files.

Compatibility fingerprints include package/backend versions, candidate schema,
typed configuration, datasets, adapter/runtime identity, and user-provided
dimensions. Resume rejects incompatible or corrupt state before model/backend
calls. Completed runs return their stored normalized result without repeating
work.
