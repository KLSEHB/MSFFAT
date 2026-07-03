"""Lightweight PyTorch reproductions of the AWF LSTM and SDAE variants."""

from __future__ import annotations

import torch
from torch import nn


class AWFLSTM(nn.Module):
    """Two-layer LSTM following the architecture released with AWF."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.22,
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(hidden[-1])


class AWFSDAE(nn.Module):
    """Compact denoising autoencoder with an AWF-compatible classifier head."""

    def __init__(self, num_classes: int = 100, input_dim: int = 5000):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, input_dim),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))

    def reconstruct(self, x: torch.Tensor, noise_probability: float = 0.15) -> torch.Tensor:
        flattened = x.flatten(1)
        if noise_probability > 0:
            keep = torch.rand_like(flattened).ge(noise_probability)
            flattened = flattened * keep
        return self.decoder(self.encoder(flattened))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))
