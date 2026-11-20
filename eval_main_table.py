#!/usr/bin/env python3
"""Evaluate the released checkpoints and reproduce the main table."""
import argparse, json
from pathlib import Path

import torch

from icb.models.registry import build_model
from icb.protocol import evaluate, load_npz, normalize_all

ROOT = Path(__file__).resolve().parent

# entries where the table prints a value rounded away from the checkpoint's exact
# result; they are accepted at a looser tolerance and reported explicitly
ROUNDING = {("pmsm_l60_h60", "lstm"): "table 0.00036, checkpoint 0.000355"}


def build_and_load(rec, data, device):
    y = data["test"]["y_hist"]
    d_y, L = int(y.shape[-1]), int(y.shape[1])
    d_u = int(data["test"]["u_hist"].shape[-1])
    H = int(data["test"]["y_future"].shape[1])
    model = build_model(rec["model"], d_y, d_u, L, H, **rec["model_config"]).to(device)
    state = torch.load(ROOT / rec["checkpoint"], map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="substring filter on dataset/model")
    args = ap.parse_args()

    runs = json.loads((ROOT / "configs" / "main_table_runs.json").read_text())

    rows, n_ok, skipped = [], 0, 0
    cache = {}
    for rec in runs:
        if True:
            ds, mdl = rec["dataset"], rec["model"]
            entry = rec.get("table_entry") or {}
            tag = f"{ds}/{mdl}"
            if args.only and args.only not in tag:
                continue
            if not rec.get("checkpoint"):
                print("%-42s SKIP (config only, no checkpoint released)" % tag, flush=True)
                skipped += 1
                continue
            if ds not in cache:
                npz = ROOT / "data" / f"{ds}.npz"
                if not npz.exists():
                    print("%-42s SKIP (data not included in this release)" % tag, flush=True)
                    skipped += 1
                    continue
                raw, stats, _ = load_npz(str(npz), device=args.device)
                cache[ds] = normalize_all(raw, stats)
            data = cache.get(ds)
            if data is None:
                skipped += 1
                continue
            if not (ROOT / rec["checkpoint"]).exists():
                print("%-42s SKIP (checkpoint in checkpoints_full.zip)" % tag, flush=True)
                skipped += 1
                continue

            torch.manual_seed(0)
            model = build_and_load(rec, data, args.device)
            m = evaluate(model, data["test"])
            want = entry.get("mse_normalized")
            got = m["mse_normalized"]
            rel = abs(got - want) / max(abs(want), 1e-12)
            # KITE draws Monte-Carlo samples at inference time; the other models
            # are deterministic.  TokaMark aside, the table prints five decimals.
            tol = 1e-2 if mdl == "kite" else 2e-3
            if (ds, mdl) in ROUNDING:
                tol = 2e-2
            ok = (round(got, 5) == round(want, 5)) or rel < tol
            n_ok += ok
            rows.append((tag, got, want, rel, ok))
            print("%-42s mse_norm=%.6g (table %.6g, rel %.1e) %s"
                  % (tag, got, want, rel, "OK" if ok else "MISMATCH"), flush=True)

    print("\n%d/%d evaluated entries match the table." % (n_ok, len(rows)))
    if skipped:
        print("%d entries skipped: checkpoints in checkpoints_full.zip, or "
              "config-only datasets (TokaMark)." % skipped)
    for (ds, mdl), why in ROUNDING.items():
        print("note: %s/%s %s" % (ds, mdl, why))
    if rows and n_ok != len(rows):
        print("Mismatching entries:")
        for tag, got, want, rel, ok in rows:
            if not ok:
                print("  %-42s %.6g vs %.6g" % (tag, got, want))


if __name__ == "__main__":
    main()
