import numpy as np
import torch

from tricg.constants import DEVICES
from tricg.types import BenchmarkCoverageMatrix, CoverageProfile, ItemConceptVector, SAEActivations


class CoverageAggregator:
    """Aggregate activations using the paper's token pooling and relative coverage."""

    def aggregate_item(self, activations: SAEActivations) -> ItemConceptVector:
        """Average selected tokens while retaining every SAE feature."""
    
        values = activations.values
        mask = activations.token_mask
        selected = values[mask.to(values.device)]
        vector = selected.detach().to(device = DEVICES.CPU, dtype = torch.float64).mean(dim = 0).numpy()
        
        return ItemConceptVector(
            item = activations.tokenized_item.item,
            values = vector,
            token_count = selected.shape[0],
            extraction = activations.extraction,
        )

    def aggregate_benchmark(self, items: list[ItemConceptVector]) -> CoverageProfile:
        """Sum item means and divide by their mean across all dictionary concepts."""
        
        first = items[0]
        totals = np.stack([item.values for item in items]).sum(axis = 0, dtype = np.float64)
        mean = totals.mean()
        
        coverage = totals / mean if mean > 0 else np.zeros_like(totals)
        
        return CoverageProfile(
            benchmark_id = first.item.benchmark_id,
            values = coverage,
            item_count = len(items),
            extraction = first.extraction,
        )

    def build_matrix(self, profiles: list[CoverageProfile]) -> BenchmarkCoverageMatrix:
        """Stack compatible profiles in the supplied benchmark order."""
        
        first = profiles[0]
        benchmark_ids = [profile.benchmark_id for profile in profiles]
       
        return BenchmarkCoverageMatrix(
            benchmark_ids = benchmark_ids,
            values = np.stack([profile.values for profile in profiles]),
            extraction = first.extraction,
        )
