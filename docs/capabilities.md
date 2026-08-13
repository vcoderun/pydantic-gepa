# Capability Map

| Capability | Common API | Advanced API | Example |
| --- | --- | --- | --- |
| Typed examples | `Example` | Pydantic Evals cases | `basic.py` |
| Explicit train/validation | `optimize` | `DataSplit` and rescore | `basic.py` |
| Instruction optimization | `Component` + injection | custom candidate scope | `basic.py` |
| Multiple components | `ComponentCatalog` | component selectors | `dot_optimization.py` |
| Tool descriptions | schema collectors | schema path utilities | `schema_components.py` |
| Output field descriptions | `ModelOutputInjection` | model schema application | `model_schema_components.py` |
| Multimodal examples | typed inputs + `Attachment` | custom evidence encoder | `dot_optimization.py` |
| Scalar scoring | score callable | Pydantic Evals evaluator | `basic.py` |
| Multi-metric evidence | `MetricResult` mapping | objective selection | `evaluation_strategies.py` |
| Controlled repeated calls | `Evaluation.controlled` | trace capture | `evaluation_strategies.py` |
| Caching | `EvaluationConfig` | `CacheStore` | API reference |
| Typed GEPA settings | `GEPAConfig` | legacy mapper | `basic.py` |
| Reflection model adapter | model id or callable | Pydantic AI adapter | `basic.py` |
| Candidate lineage | normalized result | candidate tree artifacts | result model |
| Pareto evidence | normalized result | backend frontier controls | result model |
| Staged optimization | `Plan` and `Stage` | custom stage runner | `staged_grouped.py` |
| Global and stage budgets | `Budget` | custom stop policy | `staged_grouped.py` |
| Checkpoint/resume | `RunConfig` | state store contracts | `checkpoint_resume.py` |
| Typed events | observers | backend callback bridge | `events_progress.py` |
| Rich progress | observer/config | custom observer | `events_progress.py` |
| Logfire | optional observer | reflection records | `logfire_observer.py` |
| CLI | Python target | custom Click integration | command-line docs |
| Optimize Anything engines | `Engine.gepa/autoresearch/meta_harness/best_of_n/custom` | exact custom engine contract | `experimental_optimize_anything.py` |
| Engine composition | `Sequential`, `Parallel`, `BestOf`, `Vote`, `AdaptiveSequential` | normalized selection and budget evidence | `experimental_optimize_anything.py` |
| Omni pipeline | `Pipeline` | step checkpoint and continuation lineage | `experimental_optimize_anything.py` |
| External evidence recording | recorder hook | adapter report envelopes | `recorder_hook.py` |

## Deliberate non-features

- no production prompt deployment
- no automatic source-file overwrite
- no cloud control plane requirement
- no pydantic-gepa YAML optimization DSL
- no assumption that every candidate can be safely mixed
- no claim of causal attribution from correlated candidate changes
