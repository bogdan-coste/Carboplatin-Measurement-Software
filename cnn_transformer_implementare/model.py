from __future__ import annotations

import torch
from torch import nn


class CNNTransformer(nn.Module):

    def __init__(
        self,
        input_length: int,
        cnn_channels: tuple[int, ...] = (16, 32, 64),
        embed_dim: int = 64,
        num_heads: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []
        in_channels = 1
        for out_channels in cnn_channels:

            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        reduced_length = input_length

        for _ in cnn_channels:
            reduced_length //= 2
        if reduced_length < 1:
            raise ValueError("Signal is too short for the requested CNN depth.")

        self.proj = nn.Linear(cnn_channels[-1], embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, reduced_length + 1, embed_dim) * 0.02)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout = dropout,
            batch_first = True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.proj(x)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.norm(x)
        return self.prediction_head(x[:, 0, :]).squeeze(-1)


def augment_batch(
    features: torch.Tensor, targets: torch.Tensor, noise: float = 0.03
) -> tuple[torch.Tensor, torch.Tensor]:

    scale = torch.empty(features.size(0), 1, 1, device=features.device).uniform_(0.95, 1.05)

    augmented = features * scale + torch.randn_like(features) * noise * features.abs().mean()

    return augmented, targets
