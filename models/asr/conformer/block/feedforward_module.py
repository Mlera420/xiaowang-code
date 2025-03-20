import torch.nn as nn
from torch import Tensor

from ..activate.swich import Swish

class FeedForwardModule(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.encoder_dim = encoder_dim
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, input: Tensor) -> Tensor:
        assert self.encoder_dim == input.size(-1), f"Input size {input.size(-1)} should be equal to encoder_dim {self.encoder_dim}"
        return self.sequential(input)