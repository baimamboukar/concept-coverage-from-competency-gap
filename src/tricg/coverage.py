"""Aggregate token-mean feature activations in a shared concept space."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def _check_instrument(instrument: dict) -> None:
    if not isinstance(instrument, dict):
        raise TypeError("Instrument must be a metadata mapping")
    for field in ("model", "sae", "token_policy"):
        if not isinstance(instrument.get(field), str) or not instrument[field].strip():
            raise ValueError(f"Instrument requires a nonempty {field}")
    layer = instrument.get("layer")
    if type(layer) is not int or layer < 0:
        raise ValueError("Instrument layer must be a nonnegative integer")


def build_profile(
    benchmark: str,
    activations: Iterable[Mapping],
    feature_count: int,
    instrument: dict,
) -> dict:
    """Build relative coverage from sparse token means for each benchmark item.

    Feature identifiers are zero-based dictionary coordinates. The full dictionary
    width must include silent features. Coverage sums to that width.
    """
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError("Benchmark must have a nonempty identifier")
    if type(feature_count) is not int or feature_count < 1:
        raise ValueError("Feature count must be a positive integer")
    _check_instrument(instrument)
    totals = np.zeros(feature_count, dtype=np.float64)
    item_count = 0
    for item in activations:
        if not isinstance(item, Mapping):
            raise TypeError("Each item must map feature identifiers to token means")
        seen = set()
        for feature, value in item.items():
            index = int(feature)
            if str(index) != str(feature) or not 0 <= index < feature_count:
                raise ValueError(f"Feature identifier {feature!r} is outside the dictionary")
            if index in seen:
                raise ValueError(f"Duplicate feature identifier {index}")
            seen.add(index)
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError("Activations must be finite and nonnegative")
            totals[index] += value
        item_count += 1
    mass = totals.sum()
    if item_count == 0 or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Coverage requires at least one item and positive finite activation mass")
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "instrument": dict(instrument),
        "item_count": item_count,
        "feature_count": feature_count,
        "coverage": (totals / mass * feature_count).tolist(),
    }


def build_matrix(profiles: Sequence[dict]) -> dict:
    """Stack comparable profiles with rows in the supplied benchmark order."""
    if not profiles:
        raise ValueError("At least one benchmark profile is required")
    instrument = profiles[0]["instrument"]
    feature_count = profiles[0]["feature_count"]
    _check_instrument(instrument)
    if type(feature_count) is not int or feature_count < 1:
        raise ValueError("Feature count must be a positive integer")
    benchmarks, rows = [], []
    for profile in profiles:
        if profile["schema_version"] != 1:
            raise ValueError("Unsupported profile schema version")
        if profile["instrument"] != instrument or profile["feature_count"] != feature_count:
            raise ValueError("Profiles must use the same instrument and dictionary width")
        benchmark = profile["benchmark"]
        if not isinstance(benchmark, str) or not benchmark.strip() or benchmark in benchmarks:
            raise ValueError("Benchmark identifiers must be nonempty and unique")
        row = np.asarray(profile["coverage"], dtype=np.float64)
        if row.shape != (feature_count,) or not np.isfinite(row).all() or (row < 0).any():
            raise ValueError("Coverage must contain one finite nonnegative value per feature")
        if not np.isclose(row.sum(), feature_count):
            raise ValueError("Relative coverage must sum to the dictionary width")
        benchmarks.append(benchmark)
        rows.append(row.tolist())
    return {
        "schema_version": 1,
        "instrument": dict(instrument),
        "feature_count": feature_count,
        "benchmarks": benchmarks,
        "concept_ids": list(range(feature_count)),
        "coverage": rows,
    }
