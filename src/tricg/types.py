from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from torch import Tensor


@dataclass
class Benchmark:
    id: str
    name: str
    domain: str
    description: str
    url: str
    revision: str
    license: str | None
    gated: bool
    languages: list[str]


@dataclass
class ExtractionConfig:
    """Shared settings that identify the model and sparse feature space."""

    model_id: str
    layer: int
    hook_name: str
    sae_release: str
    sae_id: str
    token_policy: str
    max_length: int = 1024
    model_revision: str | None = None
    sae_revision: str | None = None


@dataclass
class BenchmarkItem:
    """An identified source record retaining all benchmark-specific fields."""

    benchmark_id: str
    item_id: int
    data: dict[str, object]


@dataclass
class TokenizedItem:
    """The formatted prompt and integer token IDs and attention mask of shape [1, T]."""

    item: BenchmarkItem
    prompt: str
    input_ids: Tensor
    attention_mask: Tensor


@dataclass
class HiddenStateActivations:
    """Captured states of shape [T, d] with a boolean token selection mask of shape [T]."""

    tokenized_item: TokenizedItem
    values: Tensor
    token_mask: Tensor
    extraction: ExtractionConfig


@dataclass
class SAEActivations:
    """Sparse feature activations of shape [T, K] with the same token selection mask."""

    tokenized_item: TokenizedItem
    values: Tensor
    token_mask: Tensor
    extraction: ExtractionConfig


@dataclass
class ItemConceptVector:
    """agg activation over selected tokens with shape [K] in the SAE order."""

    item: BenchmarkItem
    values: NDArray[np.float64]
    token_count: int
    extraction: ExtractionConfig


@dataclass
class CoverageProfile:
    """Relative benchmark coverage of shape [K]."""

    benchmark_id: str
    values: NDArray[np.float64]
    item_count: int
    extraction: ExtractionConfig


@dataclass
class BenchmarkCoverageMatrix:
    """Coverage of shape [N, K] with benchmark in rows and SAE feature in columns."""
    
    benchmark_ids: list[str]
    values: NDArray[np.float64]
    extraction: ExtractionConfig
