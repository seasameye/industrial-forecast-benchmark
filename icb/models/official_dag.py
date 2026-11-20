# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class OfficialDAGAdapter(nn.Module):
    """Benchmark adapter for the vendored DAG model."""

    def __init__(
        self,
        d_y: int,
        d_u: int,
        L: int,
        H: int,
        repo: str | None = None,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 4,
        e_layers: int = 1,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        factor: int = 1,
        alpha: float = 0.2,
        beta: float = 0.1,
        use_c_exog: bool = True,
        use_t_exog: bool = True,
        use_c: bool = True,
        use_t: bool = True,
        infer_use_future: bool = True,
        loss: str = "MSE",
        **_: object,
    ):
        super().__init__()
        self.d_y = d_y
        self.d_u = d_u
        self.L = L
        self.H = H
        from .vendor_dag.dag_model import DAGModel

        if loss.upper() == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        cfg = SimpleNamespace(
            seq_len=L,
            pred_len=H,
            patch_len=patch_len,
            stride=stride,
            use_c=use_c,
            use_t=use_t,
            use_c_exog=use_c_exog,
            use_t_exog=use_t_exog,
            alpha=alpha,
            beta=beta,
            series_dim=d_y,
            infer_use_future=infer_use_future,
            enc_in=d_y + d_u,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            factor=factor,
            activation="gelu",
            criterion=criterion,
            loss=loss,
        )
        self.model = DAGModel(cfg)
        self._task_crit = criterion

    def training_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Official DAG objective: forecast loss plus causality regularizer."""
        x = torch.cat([batch["y_hist"], batch["u_hist"]], dim=-1)
        pred, causality_loss = self.model(x, batch["u_future"])
        return self._task_crit(pred, batch["y_future"]) + causality_loss

    def forward(self, y_hist: torch.Tensor, u_hist: torch.Tensor,
                u_future: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y_hist, u_hist], -1)
        out, _ = self.model(x, u_future)
        return out[:, -self.H:, :self.d_y]
