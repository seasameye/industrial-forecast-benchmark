# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import build_model
from .protocol import batches, evaluate, load_npz, normalize_all


def load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a model on an ICB NPZ file.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True,
                    help="persistence|lstm|crosslinear|tide|timexer|timemixer|patchtst|dlinear|dag|kite")
    ap.add_argument("--model_config", default=None,
                    help="JSON object passed to the model adapter.")
    ap.add_argument("--set", nargs="*", default=[],
                    help="Override model config values, e.g. d_model=128 repo=/tmp/TSLib")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lam_id", type=float, default=0.0,
                    help="History one-step identification loss for models exposing return_aux.")
    ap.add_argument("--zero_u", action="store_true",
                    help="Also evaluate with future controls zeroed.")
    ap.add_argument("--revin", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="Fraction of the training windows to use; val/test stay unchanged.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    # Keep all random sources aligned with the recorded experiment seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw, stats, meta = load_npz(args.data, device=args.device)
    data = normalize_all(raw, stats, revin=args.revin)
    if not 0 < args.train_frac <= 1:
        raise ValueError("--train_frac must be in (0, 1].")
    if args.train_frac < 1:
        n_train = data["train"]["y_hist"].shape[0]
        keep = max(1, int(round(n_train * args.train_frac)))
        gen = torch.Generator(device=data["train"]["y_hist"].device)
        gen.manual_seed(args.seed)
        idx = torch.randperm(n_train, generator=gen,
                             device=data["train"]["y_hist"].device)[:keep]
        data["train"] = {
            k: v[idx] if not k.startswith("_") and v.shape[0] == n_train else v
            for k, v in data["train"].items()
        }
    d_y = data["train"]["y_hist"].shape[-1]
    d_u = data["train"]["u_hist"].shape[-1]
    L = data["train"]["y_hist"].shape[1]
    H = data["train"]["u_future"].shape[1]
    model_kwargs = load_json(args.model_config)
    model_kwargs.update(parse_overrides(args.set))
    model = build_model(args.model, d_y=d_y, d_u=d_u, L=L, H=H,
                        **model_kwargs).to(args.device)

    model_key = args.model.lower()
    if model_key.endswith("_adapter"):
        future_control_mode = "future-control-adapter"
    elif model_key in {"dlinear", "patchtst", "persistence"}:
        future_control_mode = "history-only"
    elif model_kwargs.get("exo_mode") == "hist":
        future_control_mode = "history-only"
    else:
        future_control_mode = model_kwargs.get("exo_mode", "native-future-control")

    tag = args.tag or f"{Path(args.data).stem}_{args.model}"
    os.makedirs(args.out_dir, exist_ok=True)
    history_path = Path(args.out_dir) / f"{tag}.history.json"
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd) \
        if trainable else None
    crit = nn.MSELoss()
    best_norm, best_raw, bad, best_sd = float("inf"), float("inf"), 0, None
    history = []
    t0 = time.time()

    print(f"[{tag}] d_y={d_y} d_u={d_u} L={L} H={H} "
          f"windows={data['train']['y_hist'].shape[0]}/"
          f"{data['val']['y_hist'].shape[0]}/{data['test']['y_hist'].shape[0]}",
          flush=True)

    for ep in range(args.epochs if opt is not None else 0):
        torch.manual_seed(args.seed + 1000 + ep)
        model.train()
        losses = []
        for b in batches(data["train"], args.bs, shuffle=True):
            opt.zero_grad(set_to_none=True)
            if hasattr(model, "training_loss"):
                loss = model.training_loss(b)
            elif args.lam_id > 0.0:
                pred, aux = model(b["y_hist"], b["u_hist"], b["u_future"],
                                  return_aux=True, compute_id=True)
                id_pred = aux.get("history_identification")
                if id_pred is None:
                    raise RuntimeError("lam_id requested but model returned no history_identification")
                loss = crit(pred, b["y_future"]) + args.lam_id * crit(
                    id_pred[:, 1:], b["y_hist"][:, 1:])
            else:
                pred = model(b["y_hist"], b["u_hist"], b["u_future"])
                loss = crit(pred, b["y_future"])
            loss.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            losses.append(float(loss.detach()))

        val_metrics = evaluate(model, data["val"])
        vm_norm = val_metrics["mse_normalized"]
        vm_raw = val_metrics["mse"]
        if vm_norm < best_norm - 1e-7:
            best_norm, best_raw, bad = vm_norm, vm_raw, 0
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[{tag}] ep {ep + 1}/{args.epochs} "
              f"train={sum(losses)/max(1, len(losses)):.5f} "
              f"val_norm={vm_norm:.5f} val_raw={vm_raw:.5f} "
              f"best_norm={best_norm:.5f} bad={bad}", flush=True)
        history.append({
            "epoch": ep + 1,
            "train_loss": sum(losses) / max(1, len(losses)),
            "val_mse_normalized": vm_norm,
            "val_mse": vm_raw,
            "best_val_mse_normalized": best_norm,
            "bad_epochs": bad,
        })
        with open(history_path, "w") as f:
            json.dump({"tag": tag, "history": history}, f, indent=2)
        if bad >= args.patience:
            print(f"[{tag}] early stop", flush=True)
            break

    if opt is None:
        val_metrics = evaluate(model, data["val"])
        best_norm = val_metrics["mse_normalized"]
        best_raw = val_metrics["mse"]
    if best_sd is not None:
        model.load_state_dict(best_sd)
    res = {
        "tag": tag,
        "dataset": Path(args.data).stem,
        "model": args.model,
        "model_config": model_kwargs,
        "future_control_mode": future_control_mode,
        "normalization": {
            "statistics_fit": "training-history windows only",
            "metrics": "raw physical units; normalized metrics are also reported",
            "revin": args.revin,
        },
        "seed": args.seed,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "patience": args.patience,
            "gradient_clip": args.clip,
            "lambda_id": args.lam_id,
            "revin": args.revin,
            "train_frac": args.train_frac,
        },
        "params": sum(p.numel() for p in model.parameters()),
        "train_frac": args.train_frac,
        "train_windows": int(data["train"]["y_hist"].shape[0]),
        "selection_metric": "val_mse_normalized",
        "val_best_normalized": best_norm,
        "val_best_raw_mse": best_raw,
        "history": history,
        "history_file": str(history_path),
        "elapsed": int(time.time() - t0),
        "meta": meta,
        "test": evaluate(model, data["test"]),
    }
    if args.zero_u:
        res["test_zero_u"] = evaluate(model, data["test"], zero_u=True)
        res["control_dominance"] = res["test_zero_u"]["mse"] / max(
            res["test"]["mse"], 1e-12)

    out_json = Path(args.out_dir) / f"{tag}.json"
    out_pt = Path(args.out_dir) / f"{tag}.best.pt"
    res["checkpoint"] = str(out_pt)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    torch.save(model.state_dict(), out_pt)
    print(f"[{tag}] TEST mse={res['test']['mse']:.5f} -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
