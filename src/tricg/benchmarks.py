from datasets import load_dataset

from tricg.types import Benchmark, BenchmarkItem


class BenchmarkLoader:
    """Load identified items from one benchmark configuration and split."""

    def __init__(self, benchmark: Benchmark, split: str, configuration: str | None = None) -> None:
        self.benchmark = benchmark
        self.split = split
        self.configuration = configuration

    def load(self) -> list[BenchmarkItem]:
        """Return source records identified by their original row index."""
        dataset = load_dataset(
            path = self.benchmark.id,
            revision = self.benchmark.revision,
            name = self.configuration,
            split = self.split,
        )

        return [
            BenchmarkItem(
                benchmark_id = self.benchmark.id,
                item_id = index,
                data = dict(record),
            )
            for index, record in enumerate(dataset)
        ]


if __name__ == "__main__":
    from random import Random

    from rich import print

    from tricg.constants import (
        SMOKE_BENCHMARK,
        SMOKE_BENCHMARK_CONFIGURATION,
        SMOKE_BENCHMARK_SPLIT,
        SMOKE_SAMPLE_SEED,
        SMOKE_SAMPLE_SIZE,
    )

    loader = BenchmarkLoader(
        SMOKE_BENCHMARK,
        split = SMOKE_BENCHMARK_SPLIT,
        configuration = SMOKE_BENCHMARK_CONFIGURATION,
    )
    items = loader.load()
    for item in Random(SMOKE_SAMPLE_SEED).sample(items, k = SMOKE_SAMPLE_SIZE):
        print(item)
