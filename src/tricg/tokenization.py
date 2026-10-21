from typing import cast

from tricg.constants import BENCHMARK_PROMPT_FIELDS
from tricg.types import BenchmarkItem, ExtractionConfig, TokenizedItem


class BenchmarkTokenizer:
    """formats benchmark items and tokenizes prompts for the configured model"""

    def __init__(self, config: ExtractionConfig) -> None:

        from transformers import AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision = config.model_revision,
        )

    def tokenize(self, items: list[BenchmarkItem]) -> list[TokenizedItem]:
        """returns tokenized items in input order"""
        return [self._tokenize_item(item) for item in items]

    def _tokenize_item(self, item: BenchmarkItem) -> TokenizedItem:
        """returns prompt, token IDs and attention mask of shape [1, T]"""
        prompt = self._format_prompt(item)
        encoded = self.tokenizer(
            prompt,
            add_special_tokens = True,
            truncation = True,
            max_length = self.config.max_length,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        return TokenizedItem(
            item = item,
            prompt = prompt,
            input_ids = encoded["input_ids"],
            attention_mask = encoded["attention_mask"],
        )

    def _format_prompt(self, item: BenchmarkItem) -> str:
        """builds a prompt from the configured benchmark field"""
        field = BENCHMARK_PROMPT_FIELDS[item.benchmark_id]
        return cast(str, item.data[field])

    def _tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """maps tokenizer vocabulary pieces to their integer identifiers"""
        return self.tokenizer.convert_tokens_to_ids(tokens)
