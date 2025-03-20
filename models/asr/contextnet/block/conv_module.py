import torch.nn as nn
from typing import Tuple
from torch import Tensor
from ..activate.swich import Swish

class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        activation: bool = True,
        groups: int = 1,
        bias: bool = True,
    ):
        super(ConvModule, self).__init__()
        assert kernel_size == 5, "The convolution layer in the ContextNet model kernel size must be 5."

        if stride == 1:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=bias,
            )
        elif stride == 2:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                padding=padding,
                groups=groups,
                bias=bias,
            )

        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.activation = activation

        if self.activation:
            self.swish = Swish()

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for convolution layer.

        Args:
            **inputs** (torch.FloatTensor): Input of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        outputs, output_lengths = self.conv(inputs), self._get_sequence_lengths(input_lengths)
        outputs = self.batch_norm(outputs)

        if self.activation:
            outputs = self.swish(outputs)

        return outputs, output_lengths

    def _get_sequence_lengths(self, seq_lengths):
        return (
            seq_lengths + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
        ) // self.conv.stride[0] + 1