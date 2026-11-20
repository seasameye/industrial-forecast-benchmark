# -*- coding: utf-8 -*-
"""Minimal compatibility shim for third-party forecasting repos.

The imported DAG/KITE/TSLib modules import ``LSHSelfAttention`` at module load
time, even when the benchmarked models use FullAttention instead.  This shim
keeps those imports working without vendoring an unused dependency.
"""

from __future__ import annotations

from torch import nn


class LSHSelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise ImportError(
            "reformer_pytorch is not installed; LSHSelfAttention is unavailable."
        )
