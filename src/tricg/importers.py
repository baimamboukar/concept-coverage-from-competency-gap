"""Read the activation CSV exported by the authors' stage-one extractor."""

import csv
import json
from pathlib import Path


def stage1_activations(path: str | Path):
    """Yield token means from the CSV's raw token sums without loading a model."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"concept_activations", "num_tokens"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Stage-one CSV requires concept_activations and num_tokens columns")
        for row in reader:
            num_tokens = int(row["num_tokens"])
            if num_tokens < 1:
                raise ValueError(f"CSV row {reader.line_num} has no valid tokens")
            values = json.loads(row["concept_activations"])
            if not isinstance(values, dict):
                raise TypeError(f"CSV row {reader.line_num} must contain a feature mapping")
            yield {feature: float(value) / num_tokens for feature, value in values.items()}
