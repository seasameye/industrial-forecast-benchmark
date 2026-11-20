# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
from torch import nn


class OfficialTimeXerAdapter(nn.Module):
    """Adapter for decisionintelligence/DAG's TimeXer implementation."""

    def __init__(
        self,
        d_y: int,
        d_u: int,
        L: int,
        H: int,
        repo: str | None = None,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 1,
        n_heads: int = 4,
        patch_len: int = 4,
        dropout: float = 0.1,
        factor: int = 1,
        use_future: bool = True,
        **_: object,
    ):
        super().__init__()
        self.d_y = d_y
        self.d_u = d_u
        self.L = L
        self.H = H
        repo = repo or os.environ.get("DAG_REPO") or "third_party_repos/DAG"
        if not os.path.isdir(repo):
            raise FileNotFoundError(
                "DAG/TimeXer repository not found. Clone decisionintelligence/DAG "
                "and pass --set repo=/path/to/DAG or set DAG_REPO."
            )
        sys.path.insert(0, repo)
        from ts_benchmark.baselines.timexer.model.timexer_model import timexer_model

        cfg = SimpleNamespace(
            seq_len=L,
            pred_len=H,
            patch_len=patch_len,
            enc_in=d_y + d_u,
            series_dim=d_y,
            use_future=use_future,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            factor=factor,
            dropout=dropout,
            activation="gelu",
            freq="s",
        )
        self.model = timexer_model(cfg)

    def forward(self, y_hist: torch.Tensor, u_hist: torch.Tensor,
                u_future: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y_hist, u_hist], -1)
        return self.model(x, u_future)
