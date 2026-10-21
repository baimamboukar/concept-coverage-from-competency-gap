from transformer_lens import HookedTransformer

from tricg.constants import TOKEN_POLICIES
from tricg.types import ExtractionConfig, HiddenStateActivations, TokenizedItem


class ActivationExtractor:
    """Capture residual states at the configured model hook."""

    def __init__(self, config: ExtractionConfig, device: str = "cpu") -> None:


        self.config = config
        self.model = HookedTransformer.from_pretrained_no_processing(
            config.model_id,
            revision = config.model_revision,
            device = device,
        )

        self.model.eval()

    def extract(self, tokenized_item: TokenizedItem) -> HiddenStateActivations:
        """Return states of shape [T, d] and a selected-token mask of shape [T]."""
        import torch

        device = self.model.cfg.device
        input_ids = tokenized_item.input_ids.to(device)
        attention_mask = tokenized_item.attention_mask.to(device)
        with torch.inference_mode():
            _, cache = self.model.run_with_cache(
                input_ids,
                attention_mask = attention_mask,
                names_filter = [self.config.hook_name],
                stop_at_layer = self.config.layer + 1,
                return_type = None,
                prepend_bos = False,
            )
        token_mask = attention_mask[0].bool().clone()
        if self.config.token_policy == TOKEN_POLICIES.EXCLUDE_BOS:
            token_mask[0] = False

        return HiddenStateActivations(
            tokenized_item = tokenized_item,
            values = cache[self.config.hook_name][0].detach(),
            token_mask = token_mask,
            extraction = self.config,
        )
