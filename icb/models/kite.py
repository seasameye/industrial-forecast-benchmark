# -*- coding: utf-8 -*-
"""Benchmark adapter for the vendored KITE model."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


def _unroll_channels(batch: dict[str, torch.Tensor], d_y: int):
    B, L, _ = batch["y_hist"].shape
    H = batch["y_future"].shape[1] if "y_future" in batch else batch["u_future"].shape[1]
    return {
        "y_hist": batch["y_hist"].permute(0, 2, 1).reshape(B * d_y, L, 1),
        "u_hist": batch["u_hist"].repeat_interleave(d_y, dim=0),
        "u_future": batch["u_future"].repeat_interleave(d_y, dim=0),
        "y_future": batch["y_future"].permute(0, 2, 1).reshape(B * d_y, H, 1)
        if "y_future" in batch else None,
    }


class KITEOfficialAdapter(nn.Module):
    """KITE adapter.

    The upstream code currently trains via ``train_function`` and predicts via
    ``inference``.  This wrapper exposes both a benchmark ``forward`` and a
    ``training_loss(batch)`` hook consumed by ``icb.train``.
    """

    def __init__(self, d_y: int, d_u: int, L: int, H: int, repo: str | None = None,
                 flow_dim: int = 256, flow_depth: int = 2, flow_head: int = 4,
                 sampling_steps: int = 10, omega: float = 1.0,
                 p_uncond: float = 0.1, noise_dropout: float = 0.1,
                 rank: int = 8, mlp_ratio: float = 4.0,
                 inference_samples: int = 8, inference_chunk: int = 16):
        super().__init__()
        self.d_y = d_y
        self.d_u = d_u
        self.L = L
        self.H = H
        self.inference_samples = inference_samples
        self.inference_chunk = inference_chunk
        from .vendor_kite.kite_model import KITEModel

        cfg = SimpleNamespace(
            seq_len=L,
            horizon=H,
            input_dim=d_y + d_u,
            output_dim=d_y,
            flow_dim=flow_dim,
            flow_depth=flow_depth,
            flow_head=flow_head,
            num_sampling_steps=sampling_steps,
            omega=omega,
            noise_dropout=noise_dropout,
            p_uncond=p_uncond,
            structure_max=0.9,
            rank=rank,
            min_sigma=0.15,
            fc_type="Linear",
            rate=12,
            use_future_exog=True,
            agg_method="mean",
            aux_loss_weight=0.1,
            mlp_ratio=mlp_ratio,
            prior_level="sample",
        )
        self.model = KITEModel(cfg)

    def training_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        b = _unroll_channels(batch, self.d_y)
        return self.model.train_function(
            b["y_hist"], b["u_hist"], b["y_future"], b["u_future"]
        )

    def forward(self, y_hist: torch.Tensor, u_hist: torch.Tensor,
                u_future: torch.Tensor) -> torch.Tensor:
        preds = []
        for start in range(0, y_hist.shape[0], self.inference_chunk):
            end = min(start + self.inference_chunk, y_hist.shape[0])
            dummy_future = torch.zeros(end - start, u_future.shape[1],
                                       self.d_y, device=y_hist.device,
                                       dtype=y_hist.dtype)
            b = _unroll_channels({
                "y_hist": y_hist[start:end],
                "u_hist": u_hist[start:end],
                "u_future": u_future[start:end],
                "y_future": dummy_future,
            }, self.d_y)
            pred = self.model.inference(
                b["y_hist"], b["u_hist"], b["u_future"],
                num_samples=self.inference_samples,
            )
            preds.append(pred.reshape(end - start, self.d_y, -1).permute(0, 2, 1))
        return torch.cat(preds, 0)
