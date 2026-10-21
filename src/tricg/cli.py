import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from tricg.activations import ActivationExtractor
from tricg.benchmarks import BenchmarkLoader
from tricg.constants import BENCHMARK_CATALOG_PATH, SMOKE_SAMPLE_SIZE
from tricg.coverage import CoverageAggregator
from tricg.sae import SAEEncoder
from tricg.tokenization import BenchmarkTokenizer
from tricg.types import Benchmark, ExtractionConfig

app = typer.Typer(
    help="Concept Coverage for Benchmark Comparison and Selection."
)
console = Console()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"Could not read {path} — {error}") from error


def _load_catalog(path: Path) -> list[Benchmark]:
    try:
        return [Benchmark(**entry) for entry in _read_json(path)["benchmarks"]]
    except (KeyError, TypeError) as error:
        raise typer.BadParameter(f"Invalid benchmark catalog — {error}", param_hint="--catalog") from error


def _show_version(value: bool) -> None:
    if value:
        console.print(f"TriCG {version('tricg')}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=_show_version, is_eager=True)] = False,
) -> None:
    pass


@app.command()
def benchmarks(
    catalog: Annotated[Path, typer.Option(help="Benchmark catalog JSON file.")] = BENCHMARK_CATALOG_PATH,
) -> None:
    """List benchmarks in the catalog."""
    table = Table("Benchmark ID", "Name", "Domain", "Languages")
    for benchmark in _load_catalog(catalog):
        table.add_row(benchmark.id, benchmark.name, benchmark.domain, ", ".join(benchmark.languages) or "unknown")
    console.print(table)


@app.command()
def profile(
    benchmark_id: Annotated[str, typer.Argument(help="Hugging Face dataset ID from the catalog.")],
    config: Annotated[Path, typer.Option(help="ExtractionConfig fields as a JSON file.")],
    split: Annotated[str, typer.Option(help="Dataset split to profile.")],
    configuration: Annotated[str | None, typer.Option(help="Dataset configuration.")] = None,
    limit: Annotated[int, typer.Option(min=1, help="Maximum number of items to profile.")] = SMOKE_SAMPLE_SIZE,
    output: Annotated[Path | None, typer.Option(help="Save the coverage profile as JSON.")] = None,
    catalog: Annotated[Path, typer.Option(help="Benchmark catalog JSON file.")] = BENCHMARK_CATALOG_PATH,
) -> None:
    """Build a coverage profile using the pipeline interfaces."""
    benchmark = next((entry for entry in _load_catalog(catalog) if entry.id == benchmark_id), None)
    if benchmark is None:
        raise typer.BadParameter(f"Unknown benchmark {benchmark_id}", param_hint="benchmark_id")
    try:
        settings = ExtractionConfig(**_read_json(config))
    except TypeError as error:
        raise typer.BadParameter(f"Invalid extraction configuration — {error}", param_hint="--config") from error

    try:
        items = BenchmarkLoader(benchmark, split, configuration).load()[:limit]
        tokenized_items = BenchmarkTokenizer(settings).tokenize(items)
        extractor = ActivationExtractor(settings)
        encoder = SAEEncoder(settings)
        aggregator = CoverageAggregator()
        vectors = []
        for item in tokenized_items:
            hidden_states = extractor.extract(item)
            activations = encoder.encode(hidden_states)
            vectors.append(aggregator.aggregate_item(activations))
        result = aggregator.aggregate_benchmark(vectors)
    except NotImplementedError as error:
        console.print("[yellow]Profiling is not implemented yet. Pipeline methods remain scaffolds.[/yellow]")
        raise typer.Exit(1) from error

    payload = {
        "benchmark_id": result.benchmark_id,
        "item_count": result.item_count,
        "extraction": asdict(result.extraction),
        "values": result.values.tolist(),
    }
    if output:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        console.print(f"Saved coverage profile to {output}")
    else:
        console.print_json(data=payload)


if __name__ == "__main__":
    app()
