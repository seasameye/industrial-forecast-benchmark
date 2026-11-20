# -*- coding: utf-8 -*-
"""Ablation variants of IntegralControlMamba (Section: Ablation Study).

Both variants keep the history encoder, the token/control embedding, the
stateful future rollout and the training protocol of the full model; they
change exactly one mechanism each.

ControlIndependentMamba
    The selective state-space operators (decay A, write B, read C, and the
    effective step size delta) become independent of the future control.
    Controls still enter the model as input features through the value and
    gate paths, so this isolates "controls shape the state evolution" from
    "controls are merely additional input features".

DirectReadoutMamba
    The integral increment--level readout is replaced by a direct absolute
    level prediction.  The prediction no longer accumulates the previous
    prediction plus an increment, and the channel-wise increment/level gate
    is removed.
"""
from __future__ import annotations

from typing import List

from torch import Tensor, nn

from .imamba import IntegralControlMamba


class ControlIndependentMamba(IntegralControlMamba):
    """Full rollout, but A/B/C/delta do not depend on the control."""

    def __init__(self, *args, **kwargs):
        kwargs["control_independent"] = True
        super().__init__(*args, **kwargs)


class DirectReadoutMamba(IntegralControlMamba):
    """Full transition, but the readout predicts the level directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the increment branch, the gate and the increment feedthrough are unused
        del self.delta_head, self.mix_head, self.feed_delta

    def _readout(self, x: Tensor, c_raw: Tensor, prev_y: Tensor) -> Tensor:
        z = self.read_norm(x)
        return self.level_head(z) + self.feed_level(c_raw)


class _ZeroControlEncoder(nn.Module):
    """Replacement for c_enc that always emits a zero vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, c_raw: Tensor) -> Tensor:
        return c_raw.new_zeros(*c_raw.shape[:-1], self.d_model)


class ZeroControlTransitionMamba(IntegralControlMamba):
    """Old-style ablation: the state transition never sees the control.

    ``c_enc`` is replaced by a zero encoder, so every control-dependent term
    inside the transition (A_control, delta_c, control_gate, control_value,
    B, C) collapses to a constant.  The readout still receives the raw
    control-path features, so future controls remain available there.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_enc = _ZeroControlEncoder(self.d_model)
