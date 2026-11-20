# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
from torch import nn


class Persistence(nn.Module):
    """Repeat the last observed endogenous state over the horizon."""

    def forward(self, y_hist: torch.Tensor, u_hist: torch.Tensor,
                u_future: torch.Tensor) -> torch.Tensor:
        return y_hist[:, -1:].expand(-1, u_future.shape[1], -1)

