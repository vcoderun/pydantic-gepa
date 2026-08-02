from __future__ import annotations as _annotations

# pyright: reportMissingImports=false
import argparse
import base64
import json
import mimetypes
from pathlib import Path

import dotenv
import logfire

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ImageUrl

from pydantic_gepa import (
    AgentInstructionsInjection,
    Candidate,
    CandidateComponent,
    ComponentCatalog,
    Example,
    ModelOutputInjection,
    PydanticGEPAOptimization,
    model_field_accuracy,
)


class ExtractionInput(BaseModel):
    image: ImageUrl = Field(description="Customer image to extract structured data from.")


class ExtractionOutput(BaseModel):
    customer_name: str = Field(description="Extracted customer name")
    id: str = Field(description="Extracted customer id")
    pocket_id: str = Field(description="Extracted pocket id")


def load_image_url(path: Path) -> ImageUrl:
    logfire.debug("Loading image input", path=str(path))
    media_type, _ = mimetypes.guess_type(path)
    with path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return ImageUrl(
        url=f"data:{media_type or 'application/octet-stream'};base64,{encoded}",
        media_type=media_type,
    )


def build_examples(dataset_path: Path) -> list[Example[ExtractionInput, ExtractionOutput, None]]:
    with logfire.span("build examples", dataset_path=str(dataset_path)):
        entries = json.loads(dataset_path.read_text(encoding="utf-8"))
        project_root = dataset_path.parent.parent
        examples = [
            Example(
                name=entry["id"],
                inputs=ExtractionInput(
                    image=load_image_url((project_root / entry["image"]).resolve())
                ),
                expected_output=ExtractionOutput(
                    customer_name=entry["customer_name"],
                    id=entry["id"],
                    pocket_id=entry["pocket_id"],
                ),
            )
            for entry in entries
        ]
        logfire.info("Loaded optimization examples", count=len(examples))
        return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a multimodal structured-extraction agent with pydantic-gepa."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the dot project root containing dataset/dataset.json and PROMPT.md.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset.json. Overrides --project-root resolution.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="Path to the prompt markdown file. Overrides --project-root resolution.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional dotenv file. Defaults to <project-root>/.env when present.",
    )
    parser.add_argument(
        "--logfire",
        action="store_true",
        help="Send Pydantic AI and HTTP traces to the configured Logfire project.",
    )
    parser.add_argument(
        "--model",
        default="openrouter:google/gemini-3-flash-preview",
        help="Pydantic AI model identifier for the extraction agent.",
    )
    parser.add_argument(
        "--reflection-model",
        default="openrouter:openai/gpt-5.4-mini",
        help="GEPA reflection model identifier.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=25,
        help="Maximum GEPA metric calls.",
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=3,
        help="Reflection minibatch size passed to GEPA.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    env_path = args.env_file or args.project_root / ".env"
    if env_path.is_file():
        dotenv.load_dotenv(dotenv_path=env_path)
    logfire.configure(send_to_logfire=True if args.logfire else False)
    if args.logfire:
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx(capture_all=True)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    with logfire.span("resolve paths"):
        if args.dataset is not None and args.prompt is not None:
            dataset_path = args.dataset.expanduser().resolve()
            prompt_path = args.prompt.expanduser().resolve()
            logfire.info(
                "Using explicit dataset and prompt paths",
                dataset_path=str(dataset_path),
                prompt_path=str(prompt_path),
            )
            return dataset_path, prompt_path
        if args.project_root is None:
            raise SystemExit("Provide either --project-root or both --dataset and --prompt.")
        project_root = args.project_root.expanduser().resolve()
        dataset_path = (project_root / "dataset" / "dataset.json").resolve()
        prompt_path = (project_root / "PROMPT.md").resolve()
        logfire.info(
            "Resolved project paths",
            project_root=str(project_root),
            dataset_path=str(dataset_path),
            prompt_path=str(prompt_path),
        )
        return dataset_path, prompt_path


def run_agent(
    agent: Agent[None, ExtractionOutput],
    output_type: type[ExtractionOutput],
    sample: ExtractionInput,
) -> ExtractionOutput:
    with logfire.span("run extraction agent", output_type=output_type.__name__):
        result = agent.run_sync(
            [
                "Extract customer_name, id, and pocket_id from this image.",
                sample.image,
            ],
            output_type=output_type,
        )
        logfire.debug(
            "Agent produced structured output",
            customer_name=result.output.customer_name,
            customer_id=result.output.id,
            pocket_id=result.output.pocket_id,
        )
        return result.output


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    with logfire.span("dot optimization main"):
        dataset_path, prompt_path = resolve_paths(args)
        examples = build_examples(dataset_path)
        instructions = prompt_path.read_text(encoding="utf-8")
        logfire.info(
            "Preparing optimization pipeline",
            dataset_size=len(examples),
            model=args.model,
            reflection_model=args.reflection_model,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
        )

        agent = Agent[None, ExtractionOutput](
            model=args.model,
            instructions=instructions,
            output_type=ExtractionOutput,
        )
        instruction_component = CandidateComponent(
            name="instructions",
            initial_text=instructions,
            kind="instructions",
        )
        output_schema = ModelOutputInjection(
            ExtractionOutput,
            model_name="ExtractionOutput",
        )
        components = ComponentCatalog.from_components([instruction_component]).merge(
            output_schema.components
        )

        pipeline = PydanticGEPAOptimization.from_examples(
            examples=examples,
            task=lambda sample: run_agent(agent, output_schema.require(), sample),
            score=model_field_accuracy("customer_name", "id", "pocket_id"),
            score_key="accuracy",
            dataset_name="dot-extraction",
            injections=[
                AgentInstructionsInjection(
                    agent=agent,
                    candidate_component=instruction_component,
                ),
                output_schema,
            ],
            components=components,
            initial_candidate=Candidate(
                values=components.values(),
                metadata={"source": "dot_optimization.py"},
            ),
        )
        logfire.info("Starting GEPA optimization", backend=pipeline.backend)
        result = pipeline.optimize(
            max_metric_calls=args.max_metric_calls,
            reflection_lm=args.reflection_model,
            reflection_minibatch_size=args.reflection_minibatch_size,
            module_selector="all",
            display_progress_bar=True,
        )
        logfire.info(
            "GEPA optimization finished",
            best_score=result.best_score,
            best_validation_score=max(result.validation_scores)
            if result.validation_scores
            else None,
            candidate_count=len(result.candidates),
        )

        print("best score:", result.best_score)
        if result.validation_scores:
            print("best validation score:", max(result.validation_scores))
        print("best candidate:")
        for name, value in sorted(result.best_candidate.values.items()):
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
