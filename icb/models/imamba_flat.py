# -*- coding: utf-8 -*-
"""Flattened decoder ablation of IntegralControlMamba.

History encoding is identical to IntegralControlMamba. The future path is
collapsed into a single non-recurrent decode:
flatten(control_path_features(u_future)) + history summary → one MLP → y_future.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from .imamba import IntegralControlMamba, control_path_features


class FlattenedIntegralControlMamba(IntegralControlMamba):
    """Keep the history SSM; replace stepwise future integration with one MLP."""

    def __init__(
        self,
        d_y: int,
        d_u: int,
        H: int,
        d_model: int = 96,
        n_state: int = 8,
        n_layers: int = 2,
        max_horizon: int | None = None,
        dropout: float = 0.0,
        flat_hidden: int | None = None,
    ):
        super().__init__(
            d_y=d_y,
            d_u=d_u,
            d_model=d_model,
            n_state=n_state,
            n_layers=n_layers,
            max_horizon=max_horizon if max_horizon is not None else H,
            dropout=dropout,
        )
        self.H = int(H)
        c_in = 3 * d_u
        flat_in = self.H * c_in
        hidden = flat_hidden if flat_hidden is not None else 2 * d_model

        # Drop unused stepwise future heads to keep the ablation honest.
        del self.forecast_token
        del self.pos
        del self.read_norm
        del self.delta_head
        del self.level_head
        del self.mix_head
        del self.feed_delta
        del self.feed_level

        self.flat_control = nn.Sequential(
            nn.Linear(flat_in, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.flat_norm = nn.LayerNorm(2 * d_model)
        self.flat_decoder = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.H * d_y),
        )
        self._reset_flat()

    def _reset_flat(self) -> None:
        nn.init.zeros_(self.flat_decoder[-1].weight)
        nn.init.zeros_(self.flat_decoder[-1].bias)

    def summarize_history(
        self, y_hist: Tensor, states: List[Tensor]
    ) -> Tensor:
        """Aggregate final SSM states with the last observed y encoding."""
        state_summary = torch.stack(
            [state.mean(dim=-1) for state in states], dim=0
        ).mean(dim=0)
        return state_summary + self.y_enc(y_hist[:, -1])

    def encode_history(
        self, y_hist: Tensor, u_hist: Tensor, compute_id: bool = False
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        if compute_id:
            raise RuntimeError(
                "imamba_flat has no stepwise future readout; "
                "history identification (lam_id) is unsupported"
            )
        return super().encode_history(y_hist, u_hist, compute_id=False)

    def forward(
        self,
        y_hist: Tensor,
        u_hist: Tensor,
        u_future: Tensor,
        return_aux: bool = False,
        compute_id: bool = False,
    ):
        if u_future.shape[1] != self.H:
            raise ValueError(
                f"imamba_flat expects H={self.H}, got {u_future.shape[1]}"
            )
        if compute_id:
            raise RuntimeError(
                "imamba_flat has no stepwise future readout; "
                "history identification (lam_id) is unsupported"
            )

        states, _ = self.encode_history(y_hist, u_hist, compute_id=False)
        hist_summary = self.summarize_history(y_hist, states)

        prev_u = u_hist[:, -1]
        c_raw = control_path_features(u_future, prev=prev_u)
        flat_u = c_raw.reshape(c_raw.shape[0], -1)
        control_summary = self.flat_control(flat_u)

        fused = self.flat_norm(torch.cat([hist_summary, control_summary], dim=-1))
        delta = self.flat_decoder(fused).view(-1, self.H, self.d_y)
        forecast = y_hist[:, -1:, :] + delta

        if not return_aux:
            return forecast
        return forecast, {"history_identification": None}
