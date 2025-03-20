from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

class RNNDecoder(nn.Module):
    support_rnn_type = { 
        "rnn": nn.RNN,
        "lstm": nn.LSTM,
        "gru": nn.GRU,
    }

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        rnn_type: str = "lstm",
        dropout: float = 0.2,
    ):
        super(RNNDecoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        rnn_cell = self.support_rnn_type[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_dim, # after embedding 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        input: Tensor,
        state: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        x_enc = self.embedding(input)
        if state is None:
            output, state = self.rnn(x_enc) 
        else:
            output, state = self.rnn(x_enc, state)

        output = self.project(output)
        return output, state