#!/usr/bin/env python3
"""Train a main-table entry with the configuration recorded in this release.

Examples
--------
# one entry, full budget (writes runs/<tag>.json)
python train_main_table.py --dataset sysid_cascadedtanks --model lstm --device cuda

# a quick smoke test (one epoch)
python train_main_table.py --dataset sysid_cascadedtanks --model lstm --epochs 1 --device cuda

# every baseline of one dataset
python train_main_table.py --dataset sysid_cascadedtanks --all-models --device cuda
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "configs" / "main_table_runs.json"


def load_records():
    return json.loads(RUNS.read_text())


def select(records, dataset, model, all_models):
    if all_models:
        sel = [r for r in records if r["dataset"] == dataset]
    else:
        sel = [r for r in records if r["dataset"] == dataset and r["model"] == model]
    if not sel:
        raise SystemExit("no entry matches dataset=%r model=%r" % (dataset, model))
    return sel


def train_one(rec, args):
    cfg = rec["model_config"]
    tc = rec["training_config"]
    cfg_path = Path(tempfile.gettempdir()) / f"icb_cfg_{rec['dataset']}_{rec['model']}.json"
    cfg_path.write_text(json.dumps(cfg))

    epochs = args.epochs or tc.get("epochs") or 40
    tag = args.tag or f"{rec['dataset']}_{rec['model']}_seed0"
    cmd = [sys.executable, "-m", "icb.train",
           "--data", f"data/{rec['dataset']}.npz",
           "--model", rec["model"],
           "--model_config", str(cfg_path),
           "--epochs", str(epochs),
           "--bs", str(tc.get("batch_size", 64)),
           "--lr", str(tc.get("learning_rate", 1e-3)),
           "--wd", str(tc.get("weight_decay", 1e-4)),
           "--patience", str(tc.get("patience", 8)),
           "--seed", "0",
           "--device", args.device,
           "--out_dir", args.out_dir,
           "--tag", tag]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    out = Path(args.out_dir) / f"{tag}.json"
    if out.exists():
        res = json.loads(out.read_text())["test"]
        want = rec.get("table_entry", {}).get("mse_normalized")
        got = res["mse_normalized"]
        rel = abs(got - want) / max(abs(want), 1e-12) if want else float("nan")
        print("%-42s trained mse_norm=%.6g | table %.6g | rel %.1e"
              % (tag, got, want if want else float("nan"), rel), flush=True)
        return got, want
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="one of: nppad_sgtr_l60_h60, ventilator_l40_h40, pmsm_l60_h60, "
                         "sysid_cascadedtanks, sysid_wienerhammer, sysid_silverbox, tokamark_task_2-1")
    ap.add_argument("--model", default=None, help="baseline name, e.g. lstm, dlinear, patchtst")
    ap.add_argument("--all-models", action="store_true", help="train every baseline of the dataset")
    ap.add_argument("--epochs", type=int, default=None, help="override the epoch budget (smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if not args.all_models and not args.model:
        raise SystemExit("pass --model NAME or --all-models")

    (ROOT / args.out_dir).mkdir(parents=True, exist_ok=True)
    for rec in select(load_records(), args.dataset, args.model, args.all_models):
        train_one(rec, args)


if __name__ == "__main__":
    main()
