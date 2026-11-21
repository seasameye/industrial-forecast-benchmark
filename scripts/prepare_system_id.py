#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare nonlinear system-identification datasets.

Supports CSV, NPZ, and MAT files that expose arrays for u and y.  This covers
small nonlinearbenchmark.org datasets such as Cascaded Tanks, Silverbox, and
Wiener-Hammerstein once their raw files have been downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icb.protocol import save_windowed_npz


def load_array_file(path: Path, y_key: str, u_key: str):
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        return z[y_key], z[u_key]
    if path.suffix.lower() == ".npy":
        raise ValueError("Use NPZ with separate y/u keys, or CSV/MAT.")
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        z = loadmat(path)
        return z[y_key], z[u_key]
    # CSV: require columns named y_key and u_key, optionally comma-separated.
    import pandas as pd
    df = pd.read_csv(path)
    y_cols = [c.strip() for c in y_key.split(",")]
    u_cols = [c.strip() for c in u_key.split(",")]
    return df[y_cols].to_numpy(), df[u_cols].to_numpy()


def as_scenarios(y: np.ndarray, u: np.ndarray):
    y = np.asarray(y, dtype="float32")
    u = np.asarray(u, dtype="float32")
    if y.ndim == 1:
        y = y[:, None]
    if u.ndim == 1:
        u = u[:, None]
    if y.ndim == 2:
        return [(y, u, "run_0")]
    if y.ndim == 3:
        return [(y[i], u[i], f"run_{i}") for i in range(y.shape[0])]
    raise ValueError(f"unsupported y shape {y.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--y_key", default="y")
    ap.add_argument("--u_key", default="u")
    ap.add_argument("--y_names", default="")
    ap.add_argument("--u_names", default="")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="nonlinear_system_id")
    args = ap.parse_args()

    y, u = load_array_file(Path(args.file), args.y_key, args.u_key)
    scenarios = as_scenarios(y, u)
    d_y, d_u = scenarios[0][0].shape[-1], scenarios[0][1].shape[-1]
    y_names = args.y_names.split(",") if args.y_names else [f"y{i}" for i in range(d_y)]
    u_names = args.u_names.split(",") if args.u_names else [f"u{i}" for i in range(d_u)]
    save_windowed_npz(
        args.out, scenarios, y_names, u_names, args.L, args.H, args.stride,
        split=args.split, seed=args.seed, dataset=args.dataset,
        source=str(args.file),
        roles={
            "y": {name: "measured_target" for name in y_names},
            "u": {name: "future_control_command" for name in u_names},
        },
        future_availability="held-out input plan supplied by the system-identification protocol",
    )
    print(f"[system_id] saved {args.out} from {len(scenarios)} trajectories")


if __name__ == "__main__":
    main()
