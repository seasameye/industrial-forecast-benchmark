# -*- coding: utf-8 -*-
"""Causally masked joint-decoder ablation of IntegralControlMamba.

This variant isolates the causal (prefix-consistency) structure from decoder
capacity and feedback. It keeps the one-shot ablation's decoder unchanged ---
same layers, same parameter count --- and changes exactly one thing: when
producing the prediction for horizon step k, the control features of steps
k+1..H are replaced by a learned mask embedding.

Without the mask (``imamba_flat``) a single MLP sees the complete future
control path at once, so step 1 may depend on u_H. With the mask, step k only
ever sees u_1..u_k, while capacity, decoder parameterization, and the absence
of prediction feedback stay identical to the one-shot arm.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .imamba import control_path_features
from .imamba_flat import FlattenedIntegralControlMamba


class MaskedFlattenedIntegralControlMamba(FlattenedIntegralControlMamba):
    """One-shot decoder with a causal mask over the future control path."""

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
            H=H,
            d_model=d_model,
            n_state=n_state,
            n_layers=n_layers,
            max_horizon=max_horizon,
            dropout=dropout,
            flat_hidden=flat_hidden,
        )
        # One extra vector; the decoder towers are untouched, so the parameter
        # budget matches the one-shot arm up to 3*d_u entries.
        self.mask_token = nn.Parameter(torch.zeros(3 * d_u))

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
                f"imamba_masked expects H={self.H}, got {u_future.shape[1]}"
            )
        if compute_id:
            raise RuntimeError(
                "imamba_masked has no stepwise future readout; "
                "history identification (lam_id) is unsupported"
            )

        states, _ = self.encode_history(y_hist, u_hist, compute_id=False)
        hist_summary = self.summarize_history(y_hist, states)

        prev_u = u_hist[:, -1]
        c_raw = control_path_features(u_future, prev=prev_u)
        batch, H, _ = c_raw.shape

        steps = []
        for k in range(1, H + 1):
            ck = c_raw.clone()
            if k < H:
                ck[:, k:] = self.mask_token
            flat_u = ck.reshape(batch, -1)
            control_summary = self.flat_control(flat_u)
            fused = self.flat_norm(
                torch.cat([hist_summary, control_summary], dim=-1)
            )
            delta_k = self.flat_decoder(fused).view(batch, H, self.d_y)
            # Only the k-th output is causal for horizon step k.
            steps.append(delta_k[:, k - 1])

        delta = torch.stack(steps, dim=1)
        forecast = y_hist[:, -1:, :] + delta

        if not return_aux:
            return forecast
        return forecast, {"history_identification": None}
