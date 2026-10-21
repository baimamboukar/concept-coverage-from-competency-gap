import torch
from sae_lens import SAE

from tricg.constants import DEVICES
from tricg.types import ExtractionConfig, HiddenStateActivations, SAEActivations


class SAEEncoder:
    """Encode residual states with the configured sparse autoencoder checkpoint."""

    def __init__(self, config: ExtractionConfig, device: str = DEVICES.CPU) -> None:

        self.config = config
        self.sae = SAE.from_pretrained(
            release = config.sae_release,
            sae_id = config.sae_id,
            device = device,
        )
        self.sae.eval()
        

    def encode(self, hidden_states: HiddenStateActivations) -> SAEActivations:
        """Return feature activations of shape [T, K] with the original token mask."""
      
        
        values = hidden_states.values.to(device = self.sae.device, dtype = self.sae.dtype)
        
        with torch.inference_mode():
            activations = self.sae.encode(values)
            
        return SAEActivations(
            tokenized_item = hidden_states.tokenized_item,
            values = activations,
            token_mask = hidden_states.token_mask.to(activations.device),
            extraction = self.config,
        )
