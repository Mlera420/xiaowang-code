import torch.nn as nn
from ..activate.swich import Swish
from typing import Tuple
from torch import Tensor

class SEModule(nn.Module):
    def __init__(self, dim:int, ratio:int=8)->None:
        assert dim % ratio == 0, "Dimension should be divisible by ratio."
        self.dim = dim
        self.sequential = nn.Sequential(
            nn.Linear(dim, dim // ratio),
            Swish(),
            nn.Linear(dim // ratio, dim),
        )

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        residual = inputs
        seq_lengths = inputs.size(2)

        inputs = inputs.sum(dim=2) / input_lengths.unsqueeze(1)
        output = self.sequential(inputs)

        output = output.sigmoid().unsqueeze(2)
        output = output.repeat(1,1,seq_lengths)
        return output * residual