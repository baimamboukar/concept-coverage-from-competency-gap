from pathlib import Path
from types import SimpleNamespace

from tricg.types import Benchmark, ExtractionConfig

DEVICES = SimpleNamespace(
    CPU = "cpu",
    GPU = "cuda",
    MPS = "mps",
    AUTO = "auto"
)

MODELS = SimpleNamespace(
    GEMMA2__2B = "google/gemma-2-2b",
)

SAES = SimpleNamespace(
    GEMMA2__2B = "gemma-scope-2b-pt-res-canonical",
)

SAE_IDS = SimpleNamespace(
    GEMMA2__2B__LAYER20__16K = "layer_20/width_16k/canonical",
)

BENCHMARKS = SimpleNamespace(
    GSM8K = "openai/gsm8k",
)

TOKEN_POLICIES = SimpleNamespace(
    ALL = "all",
    EXCLUDE_BOS = "exclude_bos",
)


BENCHMARK_CATALOG_PATH = Path("data/benchmarks.json")
SMOKE_SAMPLE_SIZE = 5
SMOKE_SAMPLE_SEED = 42
SMOKE_BENCHMARK_SPLIT = "test"
SMOKE_BENCHMARK_CONFIGURATION = "main"

SMOKE_BENCHMARK = Benchmark(
    id = BENCHMARKS.GSM8K,
    name = "GSM8K",
    domain = "mathematical-reasoning",
    description = "Grade school math word problems.",
    url = f"https://huggingface.co/datasets/{BENCHMARKS.GSM8K}",
    revision = "740312add88f781978c0658806c59bc2815b9866",
    license = "MIT",
    gated = False,
    languages = ["en"],
)

SMOKE_EXTRACTION_CONFIG = ExtractionConfig(
    model_id = MODELS.GEMMA2__2B,
    layer = 20,
    hook_name = "blocks.20.hook_resid_post",
    sae_release = SAES.GEMMA2__2B,
    sae_id = SAE_IDS.GEMMA2__2B__LAYER20__16K,
    token_policy = TOKEN_POLICIES.EXCLUDE_BOS,
)

ACTIVE_MODELS = {
    MODELS.GEMMA2__2B: True,
}

SAE_CONFIGS = {
    (SAES.GEMMA2__2B, SAE_IDS.GEMMA2__2B__LAYER20__16K): {
        "active": True,
        "model_id": MODELS.GEMMA2__2B,
        "layer": 20,
        "hook_name": "blocks.20.hook_resid_post",
    },
}

BENCHMARK_PROMPT_FIELDS = {
    BENCHMARKS.GSM8K: "question",
}
