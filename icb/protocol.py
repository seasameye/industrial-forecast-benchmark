# -*- coding: utf-8 -*-
"""Shared dataset protocol for controlled forecasting benchmarks.

All datasets are converted to one compressed NPZ schema:

    {split}_{y_hist,u_hist,u_future,y_future}
    y_mean, y_std, u_mean, u_std
    meta: JSON string

Shapes:

    y_hist   [N, L, d_y]    historical endogenous measurements
    u_hist   [N, L, d_u]    historical exogenous/control signals
    u_future [N, H, d_u]    known future control path
    y_future [N, H, d_y]    target trajectory
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import Tensor


SPLITS = ("train", "val", "test")
FIELDS = ("y_hist", "u_hist", "u_future", "y_future")


@dataclass(frozen=True)
class DatasetMeta:
    y_names: list[str]
    u_names: list[str]
    L: int
    H: int
    stride: int | None = None
    dataset: str | None = None
    source: str | None = None
    split: dict | None = None
    notes: str | None = None
    roles: dict | None = None
    future_availability: str | None = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in self.__dict__.items()
                           if v is not None}, ensure_ascii=False)


def load_npz(path: str | Path, device: str | torch.device | None = None):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    data: dict[str, dict[str, Tensor]] = {}
    for split in SPLITS:
        data[split] = {
            field: torch.tensor(z[f"{split}_{field}"], dtype=torch.float32,
                                device=device)
            for field in FIELDS
        }
    stats = {
        name: torch.tensor(z[name], dtype=torch.float32, device=device)
        for name in ("y_mean", "y_std", "u_mean", "u_std")
    }
    return data, stats, meta


def normalize_split(split: Mapping[str, Tensor], stats: Mapping[str, Tensor],
                    revin: bool = False) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    if revin:
        mu = split["y_hist"].mean(dim=1, keepdim=True)
        sd = split["y_hist"].std(dim=1, keepdim=True) + 1e-4
    else:
        mu, sd = stats["y_mean"], stats["y_std"]
    out["y_hist"] = (split["y_hist"] - mu) / sd
    out["y_future"] = (split["y_future"] - mu) / sd
    out["u_hist"] = (split["u_hist"] - stats["u_mean"]) / stats["u_std"]
    out["u_future"] = (split["u_future"] - stats["u_mean"]) / stats["u_std"]
    out["_y_sd"] = sd if revin else sd.expand(split["y_hist"].shape[0], 1, -1)
    out["_y_mu"] = mu if revin else stats["y_mean"].view(1, 1, -1).expand(
        split["y_hist"].shape[0], 1, -1
    )
    return out


def normalize_all(raw: Mapping[str, Mapping[str, Tensor]],
                  stats: Mapping[str, Tensor], revin: bool = False):
    return {split: normalize_split(raw[split], stats, revin) for split in SPLITS}


def batches(split: Mapping[str, Tensor], batch_size: int,
            shuffle: bool = True) -> Iterator[dict[str, Tensor]]:
    n = split["y_hist"].shape[0]
    idx = torch.randperm(n, device=split["y_hist"].device) if shuffle else \
        torch.arange(n, device=split["y_hist"].device)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield {k: v[j] for k, v in split.items() if not k.startswith("_")}


@torch.no_grad()
def evaluate(model: torch.nn.Module, split: Mapping[str, Tensor],
             zero_u: bool = False, chunk: int = 128) -> dict:
    model.eval()
    normalized_sq = []
    normalized_abs = []
    raw_sq = []
    raw_abs = []
    n = split["y_hist"].shape[0]
    for i in range(0, n, chunk):
        uf = split["u_future"][i:i + chunk]
        if zero_u:
            uf = torch.zeros_like(uf)
        pred = model(split["y_hist"][i:i + chunk],
                     split["u_hist"][i:i + chunk], uf)
        target = split["y_future"][i:i + chunk]
        diff = pred - target
        normalized_sq.append(diff.square())
        normalized_abs.append(diff.abs())
        # normalize_split stores the exact per-sample inverse transform. This
        # keeps metrics in original physical units, including with RevIN.
        mu = split["_y_mu"][i:i + chunk]
        sd = split["_y_sd"][i:i + chunk]
        raw_diff = (pred * sd + mu) - (target * sd + mu)
        raw_sq.append(raw_diff.square())
        raw_abs.append(raw_diff.abs())
    normalized_err = torch.cat(normalized_sq, 0)
    normalized_abs_err = torch.cat(normalized_abs, 0)
    err = torch.cat(raw_sq, 0)
    abs_err = torch.cat(raw_abs, 0)
    normalized_per_step = normalized_err.mean(dim=(0, 2))
    normalized_per_step_mae = normalized_abs_err.mean(dim=(0, 2))
    per_step = err.mean(dim=(0, 2))
    per_step_mae = abs_err.mean(dim=(0, 2))
    return {
        "mse": float(err.mean()),
        "mae": float(abs_err.mean()),
        "mae_rms": float(abs_err.mean()),
        "mse_normalized": float(normalized_err.mean()),
        "mae_normalized": float(normalized_abs_err.mean()),
        "per_step": [float(v) for v in per_step],
        "per_step_mae": [float(v) for v in per_step_mae],
        "per_step_normalized": [float(v) for v in normalized_per_step],
        "per_step_mae_normalized": [float(v) for v in normalized_per_step_mae],
        "per_step_first10": float(per_step[:10].mean()),
        "per_step_last10": float(per_step[-10:].mean()),
        "mae_first10": float(per_step_mae[:10].mean()),
        "mae_last10": float(per_step_mae[-10:].mean()),
    }


def make_windows(y: np.ndarray, u: np.ndarray, L: int, H: int, stride: int):
    windows = []
    for s in range(0, y.shape[0] - L - H + 1, stride):
        windows.append((y[s:s + L], u[s:s + L],
                        u[s + L:s + L + H], y[s + L:s + L + H]))
    return windows


def iter_windows_with_start(y: np.ndarray, u: np.ndarray, L: int, H: int,
                            stride: int):
    """Yield ``(start, y_hist, u_hist, u_future, y_future)`` windows."""
    for start in range(0, y.shape[0] - L - H + 1, stride):
        yield (start, y[start:start + L], u[start:start + L],
               u[start + L:start + L + H], y[start + L:start + L + H])


def scenario_split(n_scenarios: int, ratios: Sequence[float],
                   seed: int = 0) -> dict[int, str]:
    if len(ratios) != 3:
        raise ValueError("ratios must be train/val/test")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_scenarios)
    n_train = int(ratios[0] * n_scenarios)
    n_val = int(ratios[1] * n_scenarios)
    out: dict[int, str] = {}
    out.update({int(i): "train" for i in idx[:n_train]})
    out.update({int(i): "val" for i in idx[n_train:n_train + n_val]})
    out.update({int(i): "test" for i in idx[n_train + n_val:]})
    return out


def save_windowed_npz(
    out_path: str | Path,
    scenarios: Sequence[tuple[np.ndarray, np.ndarray, str]],
    y_names: Sequence[str],
    u_names: Sequence[str],
    L: int,
    H: int,
    stride: int,
    split: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    dataset: str | None = None,
    source: str | None = None,
    notes: str | None = None,
    roles: dict | None = None,
    future_availability: str | None = None,
) -> None:
    """Write a list of scenario trajectories to the benchmark NPZ protocol.

    scenarios contains tuples ``(Y, U, scenario_id)`` with Y=[T,d_y], U=[T,d_u].
    Splitting is scenario-level to avoid window leakage.
    """
    assign = scenario_split(len(scenarios), split, seed)
    buf: MutableMapping[str, MutableMapping[str, list[np.ndarray]]] = {
        s: {f: [] for f in FIELDS} for s in SPLITS
    }
    split_ids: dict[str, list[str]] = {s: [] for s in SPLITS}
    split_starts: dict[str, list[int]] = {s: [] for s in SPLITS}
    for i, (y, u, scenario_id) in enumerate(scenarios):
        target_split = assign[i]
        for start, yh, uh, uf, yf in iter_windows_with_start(y, u, L, H, stride):
            buf[target_split]["y_hist"].append(yh)
            buf[target_split]["u_hist"].append(uh)
            buf[target_split]["u_future"].append(uf)
            buf[target_split]["y_future"].append(yf)
            split_ids[target_split].append(str(scenario_id))
            split_starts[target_split].append(int(start))

    save: dict[str, np.ndarray] = {}
    for split_name in SPLITS:
        for field in FIELDS:
            items = buf[split_name][field]
            if not items:
                shape = (0, L if "hist" in field else H,
                         len(y_names) if field.startswith("y") else len(u_names))
                arr = np.zeros(shape, dtype=np.float32)
            else:
                arr = np.stack(items).astype(np.float32)
            save[f"{split_name}_{field}"] = arr
        save[f"{split_name}_scenario_id"] = np.asarray(
            split_ids[split_name], dtype=np.str_
        )
        save[f"{split_name}_window_start"] = np.asarray(
            split_starts[split_name], dtype=np.int64
        )

    train_y = save["train_y_hist"].reshape(-1, len(y_names))
    train_u = save["train_u_hist"].reshape(-1, len(u_names))
    save["y_mean"] = train_y.mean(0).astype(np.float32)
    save["y_std"] = (train_y.std(0) + 1e-8).astype(np.float32)
    save["u_mean"] = train_u.mean(0).astype(np.float32)
    save["u_std"] = (train_u.std(0) + 1e-8).astype(np.float32)
    meta = DatasetMeta(
        y_names=list(y_names),
        u_names=list(u_names),
        L=L,
        H=H,
        stride=stride,
        dataset=dataset,
        source=source,
        split={s: len(split_ids[s]) for s in SPLITS},
        notes=notes,
        roles=roles,
        future_availability=future_availability,
    )
    save["meta"] = np.array(meta.to_json())
    np.savez_compressed(out_path, **save)
