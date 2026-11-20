# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*.json")
    args = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(args.runs)):
        with open(path) as f:
            r = json.load(f)
        test = r.get("test", {})
        rows.append((
            r.get("dataset", "-"),
            r.get("model", r.get("tag", "-")),
            r.get("params", 0),
            r.get("val_best", float("nan")),
            test.get("mse", r.get("test_mse", float("nan"))),
            test.get("mae", r.get("test_mae", float("nan"))),
            test.get("per_step_first10", r.get("per_step_first10", float("nan"))),
            test.get("per_step_last10", r.get("per_step_last10", float("nan"))),
            r.get("control_dominance"),
        ))
    print(f'{"dataset":24s} {"model":16s} {"params":>10s} {"val":>9s} '
          f'{"mse":>9s} {"mae":>9s} {"s1-10":>9s} {"sH-9:H":>9s} {"CDI":>8s}')
    for ds, model, params, val, test, mae, a, b, cdi in rows:
        cdi_s = f"x{cdi:.1f}" if cdi else "-"
        print(f"{ds[:24]:24s} {model[:16]:16s} {params:10d} "
              f"{val:9.5f} {test:9.5f} {mae:9.5f} "
              f"{a:9.5f} {b:9.5f} {cdi_s:>8s}")


if __name__ == "__main__":
    main()
