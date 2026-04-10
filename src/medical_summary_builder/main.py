"""CLI entry point for Medical Summary Builder — Sequential Agent Pipeline."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from .pipeline import Pipeline, PipelineContext
from .agents import (
    IntentAgent,
    ExtractionAgent,
    AnalysisAgent,
    ValidationAgent,
    ReportAgent,
)

load_dotenv()
console = Console()

DEFAULT_MODEL = "grok-4-fast"


@click.command()
@click.option("--pdf", required=True, type=click.Path(exists=True), help="Path to input medical PDF.")
@click.option("--template", required=True, type=click.Path(exists=True), help="Path to .docx template.")
@click.option("--output", default="output/summary.docx", show_default=True, help="Path for output .docx file.")
@click.option("--layout", default=None, help="Plain-text custom column instruction (skips interactive prompt).")
@click.option(
    "--model",
    default=None,
    help=(
        f"AI Builders Space model (default: {DEFAULT_MODEL}). "
        "Options: grok-4-fast, gemini-2.5-pro, gemini-3-flash-preview, deepseek, gpt-5, kimi-k2.5"
    ),
)
def cli(
    pdf: str,
    template: str,
    output: str,
    layout: str | None,
    model: str | None,
) -> None:
    """Generate a populated medical summary using a sequential agent pipeline.

    Pipeline: Intent → Extraction (cached) → Analysis → Validation → Report
    """
    if not os.getenv("AI_BUILDER_TOKEN"):
        console.print(
            "[bold red]Error:[/bold red] AI_BUILDER_TOKEN is not set. "
            "Copy .env.example to .env and add your token.",
            style="red",
        )
        sys.exit(1)

    output_path = _timestamped_path(Path(output))
    context = PipelineContext(
        pdf_path=Path(pdf),
        template_path=Path(template),
        output_path=output_path,
        model=model or os.getenv("DEFAULT_MODEL", DEFAULT_MODEL),
        layout_instruction=layout,
    )

    pipeline = Pipeline(agents=[
        IntentAgent(),
        ExtractionAgent(),
        AnalysisAgent(),
        ValidationAgent(),
        ReportAgent(),
    ])

    final = pipeline.run(context)

    console.rule("[bold green]Pipeline Complete")
    console.print(f"\n[bold green]Report:[/bold green] [cyan]{final.report_path}[/cyan]")

    if final.validation_issues:
        console.print(
            f"\n[yellow]Validation issues corrected:[/yellow] "
            f"{len(final.validation_issues)}"
        )
        for issue in final.validation_issues:
            console.print(f"  [dim]• {issue}[/dim]")


def _timestamped_path(path: Path) -> Path:
    """Insert a timestamp into the filename stem to avoid overwrites.

    e.g. output/summary.docx → output/summary_20260410_143022.docx
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


if __name__ == "__main__":
    cli()
