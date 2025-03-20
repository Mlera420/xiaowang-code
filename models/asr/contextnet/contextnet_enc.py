from torch import Tensor
from typing import Tuple
import torch.nn as nn
from .block.contextnet_block import ConvBlock

# contextnet 编码器
# input [b, l, c]
class ContextNet(nn.Module):
    support_models_param = {
        's':0.5,
        'm':1,
        'l':2
    }

    def __init__(self, 
                num_classes: int,
                model_size: str = 'm',
                input_dim: int = 80,
                num_layers: int = 5,
                kernel_size: int = 5,
                num_channels: int = 256, # 卷积层通道数，滤波器个数
                output_dim: int = 640,
                joint_ctc_attention: bool = False, # 是否输出计算的ctc loss，不解码
    ) -> None:
        super(ContextNet, self).__init__()
        assert model_size in ('s','m','l'), f'{model_size} is not supported.'
        alpha = self.support_models_param[model_size] 

        num_channels = int(num_channels * alpha)
        output_dim = int(output_dim * alpha)

        self.joint_ctc_attention = joint_ctc_attention
        if self.joint_ctc_attention:
            self.fc = nn.Linear(output_dim, num_classes, bias=False)
    
        self.block = ConvBlock.make_conv_blocks(input_dim, num_layers, kernel_size, num_channels, output_dim)
        

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        encoder_logits = None

        output = inputs.transpose(1, 2) # [b, l, c] -> [b, c, l]
        output_lengths = input_lengths

        for block in self.blocks:
            output, output_lengths = block(output, output_lengths)

        output = output.transpose(1, 2) # back to [b, l, c]

        if self.joint_ctc_attention:
            encoder_logits = self.fc(output).log_softmax(dim=2) # [b, l, c] -> [b, l, num_classes]

        return output, encoder_logits, output_lengths
