# Pydantic AI

Pydantic AI integration has two independent roles: the application subject and
the reflection model.

## Optimize agent instructions

```python
from pydantic_ai import Agent
from pydantic_gepa import AgentInstructionsInjection, Component, Example, optimize

agent = Agent(
    "openai:gpt-5-mini",
    instructions="Route the support request.",
    output_type=str,
)

instructions = Component(
    name="instructions",
    initial_text="Route the support request.",
    kind="instructions",
)

def run_agent(text: str) -> str:
    return agent.run_sync(text).output

result = optimize(
    train=train,
    validation=validation,
    task=run_agent,
    score=lambda ctx: float(ctx.output == ctx.expected_output),
    components=[instructions],
    injections=[
        AgentInstructionsInjection(
            agent=agent,
            candidate_component=instructions,
        )
    ],
    reflection="openai:gpt-5-mini",
    budget=50,
)
```

The injection enters `agent.override(instructions=...)` for each evaluation.
The baseline agent remains reusable after the run.

## Optimize structured output descriptions

```python
from pydantic import BaseModel, Field
from pydantic_gepa import ComponentCatalog, ModelOutputInjection

class ExtractionOutput(BaseModel):
    customer_name: str = Field(description="Customer name")
    account_id: str = Field(description="Account identifier")

output_schema = ModelOutputInjection(ExtractionOutput)
catalog = ComponentCatalog.from_components([instructions]).merge(
    output_schema.components
)

def extract(image) -> ExtractionOutput:
    return agent.run_sync(
        ["Extract the account data.", image],
        output_type=output_schema.require(),
    ).output
```

Pass both injections and the merged catalog to the optimization. The output
injection reconstructs candidate Pydantic model types internally.

## Reflection through Pydantic AI

```python
from pydantic_gepa.integrations.pydantic_ai import PydanticAIReflectionModel

reflection = PydanticAIReflectionModel.from_model(
    "openai:gpt-5-mini",
    timeout=30,
    retries=2,
    max_output_tokens=2_000,
)
```

You can also pass an existing string-output agent. Reflection records expose
request count, token usage, duration, cost when available, and normalized
errors.

## Async subjects

Pydantic-gepa recognizes awaitable task and score results. Use the ordinary
async Pydantic AI API inside an async task; do not detect coroutines manually.

## Tool optimization

Pydantic AI tool definitions can be collected through the generic tool-schema
surface described in [Schema optimization](schema-optimization.md). Candidate
application produces copied definitions; it does not modify the original tool
schema in place.

## Optional dependency boundary

Importing `pydantic_gepa` does not eagerly import Pydantic AI. Integration code
lives under `pydantic_gepa.integrations.pydantic_ai` so generic applications can
use candidates, evaluation, and orchestration without the SDK installed.
