# Candidates And Concurrency

`Candidate` is an immutable snapshot of named component values and lineage.
`Runtime` owns how that snapshot reaches the subject:

- `context_local` for pure or context-variable based subjects;
- `serialized` for reversible mutation of shared application state;
- `factory` through `Runtime.from_factory(...)` for isolated subject instances.

Candidate scopes restore state after success, exceptions, and cancellation.
Serialized runtimes reject concurrency greater than one. Factory runtimes may
run concurrently because each invocation owns its subject and cleanup.

Use `required_components` and `normalize=` to validate and canonicalize a
candidate before application. Stage active/frozen state belongs to `Stage`, not
to a second candidate model.
