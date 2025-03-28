from typing import Tuple, Optional
from torch import Tensor

import torch
import torch.nn as nn
from .feedforward_module import FeedForwardModule
from .multihead_module import MultiHeadedSelfAttentionModule
from .conv_module import ConvolutionModule

# feedforward module * 1/2 + input * 1/2
# multiheaded module * 1 + input * 1
# conv module * 1 + input * 1
# feedforward module * 1/2 + input * 1/2
# layernorm
class ResidualConnectionModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        module_factor: float = 1.0,
        input_factor: float = 1.0,
    ) -> None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
        else:
            return (self.module(inputs, mask) * self.module_factor) + (inputs * self.input_factor)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ) -> None:
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConvolutionModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)