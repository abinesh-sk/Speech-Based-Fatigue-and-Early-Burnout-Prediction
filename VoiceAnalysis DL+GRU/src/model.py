from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, n_mels: int = 80, n_classes: int = 3, hidden_size: int = 192, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        # After three poolings, time and freq are /8. Flatten freq into feature dim and run GRU over time.
        self.gru = nn.GRU(input_size=(n_mels // 8) * 96, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        x = x.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.cnn(x)     # (B, C, n_mels/4, T/4)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (B, T/4, C*f)
        out, _ = self.gru(x)  # (B, T/4, 2*hidden)
        out = out[:, -1, :]   # last timestep
        return self.head(out)


__all__ = ["CRNN"]
