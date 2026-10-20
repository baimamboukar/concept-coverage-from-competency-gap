# Coverage Data

Each benchmark item is represented by a sparse mapping from zero-based feature identifiers to activations averaged over its input tokens. Missing features have zero activation. `feature_count` is the full sparse autoencoder dictionary width, including features that never activate. See [the synthetic example](../examples/toy-benchmark.json) for the input structure.

For feature `c`, relative coverage is the sum of its item-level token means divided by the sum over all items and features, multiplied by the dictionary width. This equals normalizing the mean item activation vector. Coverage values sum to the dictionary width and can exceed one. They do not measure model competence or the probability that an item tests a capability. A benchmark with no activation mass has undefined coverage and is rejected.

The `instrument` records the model, layer, sparse autoencoder, and token policy. Use model and autoencoder identifiers pinned to revisions, and describe token selection and truncation in the token policy. Additional instrument fields are preserved. Profiles can be combined only when their instrument metadata and dictionary widths match exactly. The optional `source` field can record the dataset revision, configuration, split, sampling seed, and item identifiers. Keep the original item activation files for later pruning experiments.

## Commands

```sh
uv sync --dev
mkdir -p outputs
uv run tricg profile examples/toy-benchmark.json -o outputs/toy-profile.json
uv run tricg matrix outputs/toy-profile.json -o outputs/toy-matrix.json
```

The example profile is `[3, 0, 1, 0]`. Each matrix row is a benchmark in input order, and each column follows `concept_ids`. Matrix output records the common instrument and benchmark identifiers. Keep the individual profile files for item counts and dataset provenance.

## Existing Extraction Outputs

The authors' stage-one `*_with_activations.csv` exports contain raw token sums in `concept_activations` and token counts in `num_tokens`. The importer divides each item's sums by its own token count before aggregation. Supply a metadata JSON with `benchmark`, `feature_count`, and `instrument`, then run the following command.

```sh
uv run tricg profile metadata.json --stage1-csv benchmark_sae_with_activations.csv -o outputs/profile.json
```

The importer requires positive token counts and preserves silent dictionary coordinates. CSV exports may round activations or omit values below the extraction threshold, so the resulting profile describes the retained values. Record these settings in the instrument metadata. The original pickle field `feature_summary[c]["total_activation"]` already contains a token mean, while `avg_activation` averages over active tokens only. These fields must not receive the CSV normalization a second time.

The aggregation is independently implemented from the coverage definition in [Competency Gaps](https://competency-gaps.github.io/). The CSV adapter follows the released extractor's field semantics. The upstream extraction code is kept in its own repository.
