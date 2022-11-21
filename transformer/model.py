
import torch
import torch.nn as nn
import math
from utils.model import DoubleConv1d

# Reference: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TransformerEnc(nn.Module):
    # Constructor
    def __init__(
            self,
            dim_input,
            dim_model,
            num_heads,
            num_encoder_layers,
            dropout_p,
            dim_out,
    ):
        super().__init__()

        # INFO
        self.model_type = "TransformerEnc"
        self.dim_model = dim_model

        # LAYERS
        self.embedding_src = nn.Linear(dim_input, dim_model)
        self.bn_src = nn.BatchNorm1d(dim_model)
        self.positional_encoder_src = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        encoder_layers = nn.TransformerEncoderLayer(dim_model, num_heads, dim_feedforward=2048, dropout=dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.cnn_layers = nn.Sequential(
            DoubleConv1d(dim_model, dim_model, k=3, s=1, p=1),
            DoubleConv1d(dim_model, dim_model, k=3, s=1, p=1),
            nn.Conv1d(dim_model, dim_out, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, src):
        # Src size must be (batch_size, src sequence length, input dim)

        # Embedding and positional encoding: Out size = (batch_size, sequence length, dim_model)
        src = self.embedding_src(src)
        src = self.bn_src(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.positional_encoder_src(src)

        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)

        # Transformer blocks: Out size = (sequence length, batch_size, dim_model)
        transformer_out = self.transformer_encoder(src)

        # DoubleConv x 4 -> 1D-Convc: Out size = (batch_size, dim_model, sequence length)
        out = transformer_out.permute(1, 2, 0)
        out = self.cnn_layers(out)

        # Out size = (sequence length, batch_size, dim_model)
        return out.permute(2, 0, 1)
