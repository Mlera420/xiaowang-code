from typing import Tuple

import torch
import torch.nn as nn
from .block.conformer_block import ConformerBlock

# (SpecAug) + conv subsample * 1 + linear * 1 + dropout * 1 + conformer block * N
class Conformer(nn.Module):
    def __init__(
            self,
        num_classes: int,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 17,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        joint_ctc_attention: bool = True,
    ) -> None:
        super(Conformer, self).__init__()
        
