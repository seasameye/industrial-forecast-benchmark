# -*- coding: utf-8 -*-
"""IntegralControlMamba: causal control-path integration with selective scans.

The model is deliberately positioned between the paper argument and the local
benchmark interface:

* history writes a latent selective-SSM state;
* future controls modulate the transition, step size, input, and readout;
* the forecast head predicts increments from the previous output, so known
  future controls enter as a path integral rather than as a static feature map.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def inv_softplus(x: Tensor) -> Tensor:
    return torch.log(torch.expm1(x))


def control_path_features(u: Tensor, prev: Optional[Tensor] = None) -> Tensor:
    """Causal features of a known control path.

    u is [B,T,d_u].  The returned stream contains level, first difference, and
    prefix mean.  The first difference makes event phase explicit without
    giving the model an absolute "event at slot k" table to memorize.
    """
    if prev is None:
        first = torch.zeros_like(u[:, :1])
    else:
        first = u[:, :1] - prev[:, None]
    du = torch.cat([first, u[:, 1:] - u[:, :-1]], 1)
    denom = torch.arange(1, u.shape[1] + 1, device=u.device, dtype=u.dtype)
    mean = u.cumsum(1) / denom.view(1, -1, 1)
    return torch.cat([u, du, mean], -1)


class ControlMambaLayer(nn.Module):
    """One real-valued Mamba-style selective SSM layer."""

    def __init__(self, d_model: int, n_state: int = 8,
                 min_delta: float = 1e-3, min_decay: float = 1e-2,
                 control_independent: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_state = n_state
        self.min_delta = min_delta
        self.min_decay = min_decay
        self.control_independent = control_independent
        if control_independent:
            # constant A/B/C and step size: controls no longer modulate the
            # selective state-space operators (they still enter as features).
            self.b_const = nn.Parameter(torch.zeros(d_model, n_state))
            self.c_const = nn.Parameter(torch.zeros(d_model, n_state))
            with torch.no_grad():
                self.b_const[:, 0] = 1.0
                self.c_const[:, 0] = 1.0

        self.norm = nn.LayerNorm(d_model)
        self.value = nn.Linear(d_model, d_model)
        self.control_value = nn.Linear(d_model, d_model)
        self.control_gate = nn.Linear(d_model, d_model)
        self.interaction = nn.Linear(d_model, d_model)

        base = torch.linspace(0.03, 0.6, n_state).repeat(d_model, 1)
        self.a0 = nn.Parameter(inv_softplus(base))
        self.A_control = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.A_control.weight)

        self.delta_x = nn.Linear(d_model, d_model, bias=False)
        self.delta_c = nn.Linear(d_model, d_model, bias=False)
        self.delta_bias = nn.Parameter(torch.zeros(d_model))

        self.B = nn.Linear(d_model, d_model * n_state)
        self.C = nn.Linear(d_model, d_model * n_state)
        self.out = nn.Linear(d_model, d_model)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.B, self.C):
            nn.init.normal_(lin.weight, std=1e-2)
            nn.init.zeros_(lin.bias)
            with torch.no_grad():
                lin.bias.view(self.d_model, self.n_state)[:, 0].fill_(1.0)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch, self.d_model, self.n_state, device=device, dtype=dtype)

    def step(self, x: Tensor, c: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.norm(x)
        gate = torch.sigmoid(self.control_gate(c))
        v = self.value(z) + self.control_value(c) + self.interaction(z * gate)

        if self.control_independent:
            rate = F.softplus(self.a0) + self.min_decay
            delta = (
                F.softplus(self.delta_x(z) + self.delta_bias)
                + self.min_delta
            ).unsqueeze(-1)
            b = self.b_const.unsqueeze(0).expand(x.shape[0], -1, -1)
            cc = self.c_const.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            rate = F.softplus(self.a0 + self.A_control(c).unsqueeze(-1)) + self.min_decay
            delta = (
                F.softplus(self.delta_x(z) + self.delta_c(c) + self.delta_bias)
                + self.min_delta
            ).unsqueeze(-1)
            b = self.B(c).view(-1, self.d_model, self.n_state)
            cc = self.C(c).view(-1, self.d_model, self.n_state)
        a = -rate
        zdt = delta * a
        a_bar = torch.exp(zdt)
        gain = torch.expm1(zdt) / a
        state = a_bar * state + gain * b * v.unsqueeze(-1)
        y = (cc * state).sum(-1) / math.sqrt(self.n_state)
        x = x + self.out(y)
        x = x + self.mlp(self.mlp_norm(x))
        return x, state


class IntegralControlMamba(nn.Module):
    """Selective SSM whose readout is an explicit causal integral in y-space."""

    def __init__(self, d_y: int, d_u: int, d_model: int = 96,
                 n_state: int = 8, n_layers: int = 2, max_horizon: int = 64,
                 dropout: float = 0.0, control_independent: bool = False):
        super().__init__()
        self.d_y = d_y
        self.d_u = d_u
        self.d_model = d_model
        self.n_layers = n_layers

        c_in = 3 * d_u
        self.y_enc = nn.Sequential(
            nn.Linear(d_y, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.c_enc = nn.Sequential(
            nn.Linear(c_in, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.layers = nn.ModuleList([
            ControlMambaLayer(d_model, n_state,
                              control_independent=control_independent)
            for _ in range(n_layers)
        ])
        self.forecast_token = nn.Parameter(0.02 * torch.randn(d_model))
        self.pos = nn.Embedding(max_horizon, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)

        self.read_norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_y),
        )
        self.level_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_y),
        )
        self.mix_head = nn.Linear(d_model, d_y)
        self.feed_delta = nn.Linear(c_in, d_y, bias=False)
        self.feed_level = nn.Linear(c_in, d_y, bias=False)
        self.reset_readout()

    def reset_readout(self) -> None:
        for head in (self.delta_head[-1], self.level_head[-1]):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.feed_delta.weight)
        nn.init.zeros_(self.feed_level.weight)
        nn.init.zeros_(self.mix_head.weight)
        nn.init.constant_(self.mix_head.bias, -3.0)

    def _empty_states(self, y: Tensor) -> List[Tensor]:
        return [layer.init_state(y.shape[0], y.device, y.dtype) for layer in self.layers]

    def _transition(self, x: Tensor, c: Tensor, states: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        new_states = []
        for layer, state in zip(self.layers, states):
            x, state = layer.step(x, c, state)
            new_states.append(state)
        return x, new_states

    def _readout(self, x: Tensor, c_raw: Tensor, prev_y: Tensor) -> Tensor:
        z = self.read_norm(x)
        delta = self.delta_head(z) + self.feed_delta(c_raw)
        level = self.level_head(z) + self.feed_level(c_raw)
        mix = torch.sigmoid(self.mix_head(z))
        return (1.0 - mix) * (prev_y + delta) + mix * level

    def encode_history(self, y_hist: Tensor, u_hist: Tensor,
                       compute_id: bool = False) -> Tuple[List[Tensor], Optional[Tensor]]:
        c_raw = control_path_features(u_hist)
        c = self.c_enc(c_raw)
        states = self._empty_states(y_hist)
        id_preds = []
        prev_y = y_hist[:, 0]

        for t in range(y_hist.shape[1]):
            if compute_id:
                q = self.y_enc(prev_y) + self.forecast_token
                q, _ = self._transition(q, c[:, t], [s.clone() for s in states])
                id_preds.append(self._readout(q, c_raw[:, t], prev_y))

            x = self.y_enc(y_hist[:, t])
            _, states = self._transition(x, c[:, t], states)
            prev_y = y_hist[:, t]

        ids = torch.stack(id_preds, 1) if compute_id else None
        return states, ids

    def forward(self, y_hist: Tensor, u_hist: Tensor, u_future: Tensor,
                return_aux: bool = False, compute_id: bool = False):
        states, id_preds = self.encode_history(y_hist, u_hist, compute_id=compute_id)
        prev_u = u_hist[:, -1]
        c_raw = control_path_features(u_future, prev=prev_u)
        c = self.c_enc(c_raw)
        prev_y = y_hist[:, -1]

        preds = []
        for k in range(u_future.shape[1]):
            x = self.y_enc(prev_y) + self.forecast_token + self.pos.weight[k]
            x, states = self._transition(x, c[:, k], states)
            prev_y = self._readout(x, c_raw[:, k], prev_y)
            preds.append(prev_y)

        forecast = torch.stack(preds, 1)
        if not return_aux:
            return forecast
        return forecast, {'history_identification': id_preds}


if __name__ == '__main__':
    torch.manual_seed(0)
    m = IntegralControlMamba(5, 3, d_model=32, n_state=4, n_layers=2)
    y = torch.randn(2, 60, 5)
    u = torch.randn(2, 60, 3)
    fc, aux = m(y, u, u, return_aux=True, compute_id=True)
    loss = fc.square().mean() + aux['history_identification'][:, 1:].square().mean()
    loss.backward()
    print(tuple(fc.shape), tuple(aux['history_identification'].shape),
          sum(p.numel() for p in m.parameters()))
