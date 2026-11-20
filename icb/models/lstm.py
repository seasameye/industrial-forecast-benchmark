# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
from torch import nn


class LSTMEncDec(nn.Module):
    """Encoder-decoder LSTM baseline.

    Historical endogenous variables and historical controls initialize the
    decoder state.  The decoder then consumes the known future control path
    step by step.
    """

    def __init__(self, d_y: int, d_u: int, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.LSTM(d_y + d_u, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dec = nn.LSTM(d_u, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, d_y)

    def forward(self, y_hist: torch.Tensor, u_hist: torch.Tensor,
                u_future: torch.Tensor) -> torch.Tensor:
        _, state = self.enc(torch.cat([y_hist, u_hist], -1))
        out, _ = self.dec(u_future, state)
        return self.head(out)

