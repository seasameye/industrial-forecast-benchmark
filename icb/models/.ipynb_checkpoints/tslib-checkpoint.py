# -*- coding: utf-8 -*-
"""Adapters for Time-Series-Library style baselines.

This covers CrossLinear, TiDE, TimeXer, DAG, and any other model exposed as
``models.<Name>.Model`` in a TSLib-compatible repository.  The adapter keeps
third-party code unmodified and converts the benchmark protocol
``(y_hist, u_hist, u_future)`` to the model's four-argument forward signature.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class TSLibAdapter(nn.Module):
    """Thin wrapper around TSLib model classes.

    exo_mode:
      * ``hist``: x_enc = [u_hist, y_hist].  No future controls are visible.
      * ``future_as_channels``: x_enc = [u_hist, u_future_aligned, y_hist].
        This is the fairest static-feature injection for models without a
        native future-covariate route.
      * ``native``: y goes through x_enc; u_hist/u_future go through mark tensors.
        This is mainly for TiDE/TimeXer/TFT-like implementations that consume
        future covariates natively.
    """

    def __init__(
        self,
        model_name: str,
        d_y: int,
        d_u: int,
        L: int,
        H: int,
        repo: str | None = None,
        exo_mode: str = "hist",
        d_model: int = 256,
        d_ff: int = 1024,
        e_layers: int = 1,
        d_layers: int = 1,
        n_heads: int = 8,
        dropout: float = 0.1,
        patch_len: int = 4,
        stride: int | None = None,
        c_out_native: int = 16,
        moving_avg: int = 25,
        down_sampling_layers: int = 2,
        down_sampling_window: int = 2,
        down_sampling_method: str = "avg",
        decomp_method: str = "moving_avg",
        channel_independence: int = 1,
        top_k: int = 5,
        **extra,
    ):
        super().__init__()
        self.model_name = model_name
        self.d_y = d_y
        self.d_u = d_u
        self.L = L
        self.H = H
        self.exo_mode = exo_mode
        if exo_mode == "native":
            n_vars = d_y
        elif exo_mode == "future_as_channels":
            n_vars = d_y + 2 * d_u
        elif exo_mode == "hist":
            n_vars = d_y + d_u
        else:
            raise ValueError(f"unknown exo_mode {exo_mode}")

        cfg = SimpleNamespace(
            task_name="long_term_forecast",
            features="M",
            seq_len=L,
            label_len=L // 2,
            pred_len=H,
            enc_in=n_vars,
            dec_in=n_vars,
            c_out=n_vars,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            factor=3,
            dropout=dropout,
            embed="timeF",
            freq="s",
            activation="gelu",
            patch_len=patch_len,
            stride=stride or max(1, patch_len // 2),
            alpha=1.0,
            beta=1.0,
            use_norm=True,
            output_attention=False,
            moving_avg=moving_avg,
            down_sampling_layers=down_sampling_layers,
            down_sampling_window=down_sampling_window,
            down_sampling_method=down_sampling_method,
            decomp_method=decomp_method,
            channel_independence=channel_independence,
            top_k=top_k,
            num_class=1,
        )
        for key, value in extra.items():
            setattr(cfg, key, value)
        if exo_mode == "native" and model_name.lower() == "tide":
            cfg.c_out = c_out_native

        Model = self._load_model_class(model_name, repo)
        self.model = Model(cfg)
        self.cfg = cfg
        # Some TiDE implementations hard-code calendar feature width.  Replacing
        # this one input block lets known controls play the intended covariate
        # role while keeping the rest of the official implementation intact.
        if exo_mode == "native" and model_name.lower() == "tide" and \
                hasattr(self.model, "feature_encoder"):
            res_block = type(self.model.feature_encoder)
            self.model.feature_dim = d_u
            self.model.feature_encoder = res_block(d_u, d_model, 2, dropout, True)

    @staticmethod
    def _load_model_class(model_name: str, repo: str | None):
        if repo:
            if not os.path.isdir(repo):
                raise FileNotFoundError(
                    f"TSLib repo not found: {repo}. Clone the official repo and "
                    "pass --repo /path/to/repo."
                )
            sys.path.insert(0, repo)
            if os.path.basename(os.path.normpath(repo)) == "time_series_library":
                p = Path(repo).resolve()
                # vendored TSLib lives at <root>/ts_benchmark/baselines/time_series_library
                root = p.parents[2] if len(p.parents) >= 3 and p.parents[1].name == "ts_benchmark" else p.parent
                sys.path.insert(0, str(root))
                mod = importlib.import_module(
                    f"ts_benchmark.baselines.time_series_library.models.{model_name}"
                )
                return getattr(mod, "Model", getattr(mod, model_name))
        if not repo and model_name.lower() == "crosslinear":
            from icb.third_party.CrossLinear_model import Model
            return Model
        if not repo and model_name.lower() == "tide":
            from icb.third_party.TiDE_model import Model
            return Model
        try:
            mod = importlib.import_module(f"models.{model_name}")
            return getattr(mod, "Model", getattr(mod, model_name))
        except Exception as exc:
            raise ImportError(
                f"Could not import models.{model_name}.Model. For CrossLinear, "
                "TiDE, TimeXer, and DAG, pass a TSLib-compatible repository via "
                "--repo or model.repo in the JSON config."
            ) from exc

    def _inputs(self, y_hist: torch.Tensor, u_hist: torch.Tensor,
                u_future: torch.Tensor):
        if self.exo_mode == "native":
            x_enc = y_hist
            x_mark = u_hist
            x_dec = torch.zeros(y_hist.shape[0], self.cfg.label_len + self.H,
                                self.d_y, device=y_hist.device,
                                dtype=y_hist.dtype)
            y_mark = torch.cat([u_hist[:, -self.cfg.label_len:], u_future], 1)
            return x_enc, x_mark, x_dec, y_mark
        if self.exo_mode == "future_as_channels":
            if u_future.shape[1] == self.L:
                uf_aligned = u_future
            elif u_future.shape[1] > self.L:
                raise ValueError(
                    "future_as_channels cannot consume H>L without an explicit "
                    "alignment policy; refusing to truncate u_future. Use H<=L, "
                    "a native future-covariate adapter, or implement a documented "
                    "resampling policy."
                )
            else:
                pad = u_future[:, -1:].expand(-1, self.L - u_future.shape[1], -1)
                uf_aligned = torch.cat([u_future, pad], 1)
            x_enc = torch.cat([u_hist, uf_aligned, y_hist], -1)
        else:
            x_enc = torch.cat([u_hist, y_hist], -1)
        x_dec = torch.zeros(y_hist.shape[0], self.cfg.label_len + self.H,
                            x_enc.shape[-1], device=y_hist.device,
                            dtype=y_hist.dtype)
        return x_enc, None, x_dec, None

    def forward(self, y_hist: torch.Tensor, u_hist: torch.Tensor,
                u_future: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(y_hist, u_hist, u_future)
        out = self.model(*inputs)
        return out[:, -self.H:, -self.d_y:]
