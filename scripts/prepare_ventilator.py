#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the Google Brain Ventilator Pressure Prediction Kaggle train.csv.

The benchmark uses breath-level splits.  y = pressure; known future controls are
u_in, u_out plus repeated lung parameters R and C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icb.protocol import save_windowed_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--H", type=int, default=40)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.train_csv)
    scenarios = []
    for bid, g in df.groupby("breath_id", sort=True):
        g = g.sort_values("time_step")
        y = g[["pressure"]].to_numpy(dtype="float32")
        u = g[["u_in", "u_out", "R", "C"]].to_numpy(dtype="float32")
        scenarios.append((y, u, str(bid)))
    save_windowed_npz(
        args.out,
        scenarios,
        y_names=["pressure"],
        u_names=["u_in", "u_out", "R", "C"],
        L=args.L,
        H=args.H,
        stride=args.stride,
        split=args.split,
        seed=args.seed,
        dataset="ventilator_pressure",
        source="Google Brain Ventilator Pressure Prediction (Kaggle 2021)",
        notes="Uses train.csv only because test pressure is hidden.",
        roles={
            "y": {"pressure": "measured_target"},
            "u": {
                "u_in": "future_control_command",
                "u_out": "future_control_command",
                "R": "static_patient_covariate",
                "C": "static_patient_covariate",
            },
        },
        future_availability="u_in/u_out are supplied future controls; R/C are static covariates",
    )
    print(f"[ventilator] saved {args.out} from {len(scenarios)} breaths")


if __name__ == "__main__":
    main()
