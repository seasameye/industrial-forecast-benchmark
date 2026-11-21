#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the PMSM electric motor temperature CSV for the ICB protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icb.protocol import save_windowed_npz


DEFAULT_Y = "pm,stator_yoke,stator_tooth,stator_winding"
DEFAULT_U = "u_q,u_d,i_q,i_d,motor_speed,torque,coolant,ambient"


def parse_cols(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="configs/datasets/PMSM/measures_v2.csv")
    ap.add_argument("--out", default="data/pmsm_l60_h60.npz")
    ap.add_argument("--profile_col", default="profile_id")
    ap.add_argument("--y_cols", default=DEFAULT_Y)
    ap.add_argument("--u_cols", default=DEFAULT_U)
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--H", type=int, default=60)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    y_cols = parse_cols(args.y_cols)
    u_cols = parse_cols(args.u_cols)
    df = pd.read_csv(args.csv)
    missing = [c for c in [args.profile_col, *y_cols, *u_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {args.csv}: {missing}")

    scenarios = []
    for pid, g in df.groupby(args.profile_col, sort=True):
        y = g[y_cols].to_numpy(dtype="float32")
        u = g[u_cols].to_numpy(dtype="float32")
        scenarios.append((y, u, f"profile_{pid}"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_windowed_npz(
        args.out,
        scenarios=scenarios,
        y_names=y_cols,
        u_names=u_cols,
        L=args.L,
        H=args.H,
        stride=args.stride,
        split=args.split,
        seed=args.seed,
        dataset="pmsm",
        source=str(Path(args.csv)),
        notes=(
            "PMSM temperatures are targets. u_q/u_d are treated as commands; "
            "the remaining recorded channels are supplied future covariates "
            "and must not be described as online controls without a deployment source."
        ),
        roles={
            "y": {name: "measured_target" for name in y_cols},
            "u": {
                name: (
                    "command_or_control" if name in {"u_q", "u_d"}
                    else "supplied_future_covariate"
                )
                for name in u_cols
            },
        },
        future_availability=(
            "offline_recorded_oracle_covariate; verify online availability per channel"
        ),
    )
    print(f"[pmsm] saved {args.out} from {len(scenarios)} profiles")


if __name__ == "__main__":
    main()
