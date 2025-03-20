from torch import Tensor
from typing import Tuple
import torch.nn as nn
from ..activate.swich import Swish
from .se_module import SEModule as SE
from .conv_module import ConvModule as Conv

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 5,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        residual: bool = True,
    ) -> None:
        super(ConvBlock, self).__init__()

        self.num_layers = num_layers
        self.swish = Swish()
        self.se_layer = SE(out_channels)
        self.residual = None

        if residual:
            self.residual = Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=False,
            )

        if num_layers == 1:
            self.residual = Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

        else:
            # 除了这个卷积块的最后一层卷积是自定义的，其他层的stride都是1
            stride_list = [1 for _ in range(num_layers - 1)] + [stride]
            in_channel_list = [in_channels] + [out_channels for _ in range(num_layers - 1)]

            self.conv_layers = nn.ModuleList(list())
            for in_channel, stride in zip(in_channel_list, stride_list):
                self.conv_layers.append(
                    Conv(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
        ) -> Tuple[Tensor, Tensor]:
        if self.num_layers == 1:
                output, output_lengths = self.conv_layers(inputs, input_lengths)
        else:
            for layer in self.conv_layers:
                output, output_lengths = layer(inputs, input_lengths)

        output = self.se_layer(output, output_lengths)
        
        if self.residual is not None:
            residual, residual_lengths = self.residual(inputs, input_lengths)
            output += residual

    @staticmethod
    # 创建 23个 卷积块
    def make_conv_blocks(
        input_dim: int = 80,
        num_layers: int = 5,
        kernel_size: int = 5,
        num_channels: int = 256,
        output_dim: int = 640,
    ) -> nn.ModuleList:
        conv_blocks = nn.ModuleList()

        # C0 : 1 conv layer, init_dim output channels, stride 1, no residual
        conv_blocks.append(ConvBlock(input_dim, num_channels, 1, kernel_size, 1, 0, False))

        # C1-2 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(1, 2 + 1):
            conv_blocks.append(ConvBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))

        # C3 : 5 conv layer, init_dim output channels, stride 2
        conv_blocks.append(ConvBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, True))

        # C4-6 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(4, 6 + 1):
            conv_blocks.append(ConvBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))

        # C7 : 5 conv layers, init_dim output channels, stride 2
        conv_blocks.append(ConvBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, True))

        # C8-10 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(8, 10 + 1):
            conv_blocks.append(ConvBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))

        # C11-13 : 5 conv layers, middle_dim output channels, stride 1
        conv_blocks.append(ConvBlock(num_channels, num_channels << 1, num_layers, kernel_size, 1, 0, True))
        for _ in range(12, 13 + 1):
            conv_blocks.append(
                ConvBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, True)
            )

        # C14 : 5 conv layers, middle_dim output channels, stride 2
        conv_blocks.append(ConvBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 2, 0, True))

        # C15-21 : 5 conv layers, middle_dim output channels, stride 1
        for i in range(15, 21 + 1):
            conv_blocks.append(
                ConvBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, True)
            )

        # C22 : 1 conv layer, final_dim output channels, stride 1, no residual
        conv_blocks.append(ConvBlock(num_channels << 1, output_dim, 1, kernel_size, 1, 0, False))

        return conv_blocks
