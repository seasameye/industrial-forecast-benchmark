#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a TokaMark task into the ICB NPZ protocol (v1 mapping, streaming).

v1 channel rule (control-aware forecasting only):

    y  <- task output keys          (same columns for y_hist / y_future)
    u  <- task actuator keys
    input keys not in output       -> dropped (logged)

Tasks without actuators (group 1 reconstruction) are rejected in v1.

Memory model: shot batches are loaded, windowed, flushed to disk shards, then
freed. Trajectories are never accumulated in a giant Python list.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icb.protocol import FIELDS, SPLITS, DatasetMeta, make_windows


def _require_tokamark():
    try:
        from tokamark.tasks import get_task_config, get_task_metadata  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "tokamark is not importable. Activate tokamark-env, or use --map-only "
            "only after installing tokamark."
        ) from exc


def map_task_channels(task: str) -> dict[str, Any]:
    """Map TokaMark roles to ICB y/u names; drop extra inputs."""
    from tokamark.tasks import get_task_config, get_task_metadata

    config = get_task_config(task)
    meta = get_task_metadata(config, verbose=False)
    input_keys = list(meta.get("input", {}))
    actuator_keys = list(meta.get("actuator", {}))
    output_keys = list(meta.get("output", {}))
    dropped_inputs = [k for k in input_keys if k not in output_keys]

    if not output_keys:
        raise ValueError(f"{task}: empty output keys")
    if not actuator_keys:
        raise ValueError(
            f"{task}: no actuators (v1 only supports control-aware tasks; "
            "group-1 reconstruction is out of scope)"
        )

    seg = config.get("task_window_segmenter", {})
    return {
        "task": task,
        "y_names": output_keys,
        "u_names": actuator_keys,
        "input_keys": input_keys,
        "dropped_inputs": dropped_inputs,
        "stride_in_secs": meta.get("stride_in_secs"),
        "input_length_s": seg.get("input_length"),
        "output_length_s": seg.get("output_length"),
        "delta_s": seg.get("delta"),
        "config": config,
        "meta": meta,
    }


def flatten_time_last(values: np.ndarray) -> np.ndarray:
    """Return [T, F], assuming the last axis is time."""
    values = np.asarray(values)
    if values.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if values.ndim == 1:
        return values.astype(np.float32)[:, None]
    t = values.shape[-1]
    feat = int(np.prod(values.shape[:-1]))
    return values.reshape(feat, t).T.astype(np.float32)


def interpolate_signal(
    time: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray | None:
    """Interpolate flattened [T,F] signal onto ``grid``. Returns None if unusable."""
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    flat = flatten_time_last(values)
    if time.size < 2 or flat.shape[0] != time.size or flat.shape[1] == 0:
        return None
    order = np.argsort(time)
    time = time[order]
    flat = flat[order]
    out = np.empty((grid.size, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = np.isfinite(col)
        if good.sum() < 2:
            return None
        out[:, j] = np.interp(grid, time[good], col[good]).astype(np.float32)
    return out


def _reference_dt(mapping: dict[str, Any]) -> float:
    meta = mapping["meta"]
    for key in mapping["y_names"]:
        dt = meta["output"].get(key, {}).get("dt")
        if dt is not None and float(dt) > 0:
            return float(dt)
    for key in mapping["u_names"]:
        dt = meta["actuator"].get(key, {}).get("dt")
        if dt is not None and float(dt) > 0:
            return float(dt)
    return 1e-3


def shot_to_yu(
    shot: dict[str, Any],
    y_names: list[str],
    u_names: list[str],
    dt: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align one MastDataset shot onto a common time grid → Y[T,d_y], U[T,d_u]."""
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    t_lo, t_hi = -np.inf, np.inf
    for name in y_names + u_names:
        item = shot.get(name)
        if not item:
            return None
        time = np.asarray(item.get("time", []), dtype=np.float64).reshape(-1)
        values = item.get("values")
        if time.size < 2 or values is None:
            return None
        series[name] = (time, np.asarray(values))
        t_lo = max(t_lo, float(time.min()))
        t_hi = min(t_hi, float(time.max()))
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
        return None
    n = int(np.floor((t_hi - t_lo) / dt)) + 1
    if n < 2:
        return None
    grid = t_lo + np.arange(n, dtype=np.float64) * dt

    y_cols, u_cols = [], []
    for name in y_names:
        arr = interpolate_signal(*series[name], grid)
        if arr is None:
            return None
        y_cols.append(arr)
    for name in u_names:
        arr = interpolate_signal(*series[name], grid)
        if arr is None:
            return None
        u_cols.append(arr)
    y = np.concatenate(y_cols, axis=1)
    u = np.concatenate(u_cols, axis=1)
    if not np.isfinite(y).all() or not np.isfinite(u).all():
        return None
    return y.astype(np.float32), u.astype(np.float32)


def expand_feature_names(
    mapping: dict[str, Any],
    sample_shot: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Expand multi-channel signals into flat feature names from one sample shot."""
    y_names, u_names = [], []
    for role, keys, out in (
        ("y", mapping["y_names"], y_names),
        ("u", mapping["u_names"], u_names),
    ):
        del role
        for key in keys:
            values = sample_shot.get(key, {}).get("values")
            if values is None:
                out.append(key)
                continue
            flat = flatten_time_last(np.asarray(values))
            width = flat.shape[1] if flat.ndim == 2 else 1
            if width <= 1:
                out.append(key)
            else:
                out.extend(f"{key}#{i}" for i in range(width))
    return y_names, u_names


def list_local_shot_ids(zarr_root: str | Path) -> set[int]:
    """Shot IDs present under a local zarr root (dirs named ``12345`` or ``12345.zarr``)."""
    root = Path(zarr_root)
    ids: set[int] = set()
    for p in root.iterdir():
        name = p.name
        if name.endswith(".zarr"):
            name = name[: -len(".zarr")]
        if name.isdigit():
            ids.add(int(name))
    return ids


def required_signals_dict(config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Build MAST availability filter from actuator+output only (v1 y/u)."""
    out: dict[str, list[str]] = {}
    src = config.get("sources_and_signals", {})
    for role in ("actuator_name", "output_name"):
        for source, signal in (src.get(role) or []):
            out.setdefault(source, [])
            if signal not in out[source]:
                out[source].append(signal)
    return out


def filter_shots_by_availability(
    shots: list[int],
    *,
    config: Mapping[str, Any],
    zarr_root: str | None,
) -> list[int]:
    """Keep only shots that report all required y/u signals in the availability CSV."""
    from MAST_tools.utils.store_utils import MASTStorageManager

    req = required_signals_dict(config)
    kwargs = {}
    if zarr_root:
        kwargs["base_local_zarr_path"] = zarr_root
    sm = MASTStorageManager(**kwargs)
    try:
        ok = set(int(s) for s in sm.list_shots_by_signal_availability(req))
    except Exception as exc:
        print(f"[tokamark] availability filter unavailable ({exc}); "
              "falling back to unfiltered split list")
        return shots
    keep = [s for s in shots if int(s) in ok]
    print(f"[tokamark] availability filter: {len(shots)} -> {len(keep)} "
          f"(need sources {list(req)})")
    return keep


class StreamingWindowWriter:
    """Window trajectories and flush to disk shards; finalize into one ICB NPZ."""

    def __init__(
        self,
        shard_dir: Path,
        *,
        L: int,
        H: int,
        stride: int,
        flush_windows: int = 256,
    ):
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.L = L
        self.H = H
        self.stride = stride
        self.flush_windows = max(1, flush_windows)
        self.buf: dict[str, dict[str, list[np.ndarray]]] = {
            s: {f: [] for f in FIELDS} for s in SPLITS
        }
        self.buf_ids: dict[str, list[str]] = {s: [] for s in SPLITS}
        self.counts = {s: 0 for s in SPLITS}
        self.shard_idx = 0
        self.kept_shots = 0
        # Running train stats from windowed y_hist / u_hist (for ICB normalize).
        self._y_sum: np.ndarray | None = None
        self._y_sq: np.ndarray | None = None
        self._u_sum: np.ndarray | None = None
        self._u_sq: np.ndarray | None = None
        self._n_y = 0
        self._n_u = 0

    def _pending(self) -> int:
        return sum(len(self.buf[s]["y_hist"]) for s in SPLITS)

    def add_shot(self, split: str, y: np.ndarray, u: np.ndarray, shot_id: str) -> int:
        if y.shape[0] < self.L + self.H or u.shape[0] < self.L + self.H:
            return 0
        windows = make_windows(y, u, self.L, self.H, self.stride)
        if not windows:
            return 0
        n_new = 0
        for yh, uh, uf, yf in windows:
            self.buf[split]["y_hist"].append(np.asarray(yh, dtype=np.float32))
            self.buf[split]["u_hist"].append(np.asarray(uh, dtype=np.float32))
            self.buf[split]["u_future"].append(np.asarray(uf, dtype=np.float32))
            self.buf[split]["y_future"].append(np.asarray(yf, dtype=np.float32))
            self.buf_ids[split].append(str(shot_id))
            n_new += 1
            if split == "train":
                self._update_stats(yh, uh)
        self.kept_shots += 1
        self.counts[split] += n_new
        if self._pending() >= self.flush_windows:
            self.flush()
        return n_new

    def _update_stats(self, yh: np.ndarray, uh: np.ndarray) -> None:
        yh2 = yh.reshape(-1, yh.shape[-1]).astype(np.float64)
        uh2 = uh.reshape(-1, uh.shape[-1]).astype(np.float64)
        if self._y_sum is None:
            self._y_sum = yh2.sum(0)
            self._y_sq = (yh2 ** 2).sum(0)
            self._u_sum = uh2.sum(0)
            self._u_sq = (uh2 ** 2).sum(0)
        else:
            self._y_sum += yh2.sum(0)
            self._y_sq += (yh2 ** 2).sum(0)
            self._u_sum += uh2.sum(0)
            self._u_sq += (uh2 ** 2).sum(0)
        self._n_y += yh2.shape[0]
        self._n_u += uh2.shape[0]

    def flush(self) -> None:
        if self._pending() == 0:
            return
        payload: dict[str, Any] = {}
        for split in SPLITS:
            for field in FIELDS:
                items = self.buf[split][field]
                key = f"{split}_{field}"
                if items:
                    payload[key] = np.stack(items).astype(np.float32)
                self.buf[split][field] = []
            self.buf_ids[split] = []
        path = self.shard_dir / f"shard_{self.shard_idx:05d}.npz"
        np.savez(path, **payload)  # uncompressed: faster flush, less RAM spike
        self.shard_idx += 1
        del payload
        gc.collect()

    def finalize(
        self,
        out_path: Path,
        *,
        y_names: list[str],
        u_names: list[str],
        dataset: str | None,
        source: str | None,
        notes: str | None,
    ) -> None:
        self.flush()
        shards = sorted(self.shard_dir.glob("shard_*.npz"))
        if not shards:
            raise RuntimeError("no window shards written; check L/H vs shot lengths")

        d_y, d_u = len(y_names), len(u_names)
        totals = {s: 0 for s in SPLITS}
        for path in shards:
            z = np.load(path)
            for split in SPLITS:
                key = f"{split}_y_hist"
                if key in z.files:
                    totals[split] += int(z[key].shape[0])
            z.close()

        # Disk-backed final arrays (avoid keeping every shard in RAM).
        mmap_dir = self.shard_dir / "_mmap"
        if mmap_dir.exists():
            shutil.rmtree(mmap_dir)
        mmap_dir.mkdir(parents=True, exist_ok=True)
        mmaps: dict[str, np.memmap] = {}
        cursors = {s: 0 for s in SPLITS}
        for split in SPLITS:
            n = totals[split]
            for field in FIELDS:
                tlen = self.L if "hist" in field else self.H
                width = d_y if field.startswith("y") else d_u
                key = f"{split}_{field}"
                shape = (n, tlen, width) if n > 0 else (0, tlen, width)
                mmaps[key] = np.lib.format.open_memmap(
                    mmap_dir / f"{key}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                )

        for path in shards:
            z = np.load(path)
            for split in SPLITS:
                ykey = f"{split}_y_hist"
                if ykey not in z.files or z[ykey].shape[0] == 0:
                    continue
                n = int(z[ykey].shape[0])
                i0 = cursors[split]
                i1 = i0 + n
                for field in FIELDS:
                    key = f"{split}_{field}"
                    mmaps[key][i0:i1] = z[key]
                cursors[split] = i1
            z.close()
            del z
            gc.collect()

        if self._n_y == 0 or self._y_sum is None:
            raise RuntimeError("no train windows; cannot compute y/u stats")
        y_mean = (self._y_sum / self._n_y).astype(np.float32)
        u_mean = (self._u_sum / self._n_u).astype(np.float32)
        y_std = (np.sqrt(np.maximum(self._y_sq / self._n_y - y_mean.astype(np.float64) ** 2, 0.0))
                 + 1e-8).astype(np.float32)
        u_std = (np.sqrt(np.maximum(self._u_sq / self._n_u - u_mean.astype(np.float64) ** 2, 0.0))
                 + 1e-8).astype(np.float32)

        meta = DatasetMeta(
            y_names=list(y_names),
            u_names=list(u_names),
            L=self.L,
            H=self.H,
            stride=self.stride,
            dataset=dataset,
            source=source,
            split={s: totals[s] for s in SPLITS},
            notes=notes,
        )
        extras = {
            "y_mean": y_mean,
            "y_std": y_std,
            "u_mean": u_mean,
            "u_std": u_std,
            "meta": np.array(meta.to_json()),
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # One array at a time into an uncompressed NPZ (ZIP_STORED) so peak RAM
        # stays near one field, not the whole dataset.
        import zipfile
        from numpy.lib.format import write_array

        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_out.exists():
            tmp_out.unlink()
        with zipfile.ZipFile(tmp_out, mode="w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as zf:
            for key, mm in mmaps.items():
                with zf.open(f"{key}.npy", mode="w", force_zip64=True) as dest:
                    write_array(dest, mm, allow_pickle=False)
            for key, arr in extras.items():
                with zf.open(f"{key}.npy", mode="w", force_zip64=True) as dest:
                    write_array(dest, np.asarray(arr), allow_pickle=False)
        for v in mmaps.values():
            del v
        mmaps.clear()
        gc.collect()
        shutil.rmtree(mmap_dir, ignore_errors=True)
        tmp_out.replace(out_path)
        print(f"[tokamark] windows={totals} shots_kept={self.kept_shots} "
              f"shards={len(shards)} -> {out_path}")

def export_streaming(
    mapping: dict[str, Any],
    *,
    out_path: Path,
    L: int,
    H: int,
    stride: int,
    local: bool,
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
    use_std_scaling: bool,
    zarr_root: str | None,
    batch_size: int,
    flush_windows: int,
    keep_shards: bool,
) -> None:
    from tokamark.data import initialize_MAST_dataset
    from tokamark.data_split import get_train_test_val_shots

    train_shots, test_shots, val_shots = get_train_test_val_shots(
        max_index_for_train=max_train,
        max_index_for_val=max_val,
        max_index_for_test=max_test,
    )
    split_lists = [
        ("train", list(train_shots)),
        ("val", list(val_shots)),
        ("test", list(test_shots)),
    ]

    store_manager_settings = None
    if zarr_root:
        zarr_root = str(Path(zarr_root).expanduser().resolve())
        if not Path(zarr_root).is_dir():
            raise SystemExit(f"--zarr-root not found: {zarr_root}")
        store_manager_settings = {"base_local_zarr_path": zarr_root}
        local_ids = list_local_shot_ids(zarr_root)
        print(f"[tokamark] local zarr root: {zarr_root} ({len(local_ids)} shots on disk)")
        if local and local_ids:
            filtered = []
            for split_name, shots in split_lists:
                keep = [s for s in shots if int(s) in local_ids]
                print(f"[tokamark] {split_name}: split={len(shots)} on_disk={len(keep)}")
                filtered.append((split_name, keep))
            split_lists = filtered

    split_lists = [
        (name, filter_shots_by_availability(
            shots, config=mapping["config"], zarr_root=zarr_root))
        for name, shots in split_lists
    ]

    dt = _reference_dt(mapping)
    shard_dir = Path(str(out_path) + ".shards")
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    writer = StreamingWindowWriter(
        shard_dir, L=L, H=H, stride=stride, flush_windows=flush_windows,
    )

    feat_y_names: list[str] | None = None
    feat_u_names: list[str] | None = None
    skipped = 0
    load_errors_shown = 0
    n_windows = 0

    for split_name, shots in split_lists:
        if not shots:
            continue
        print(f"[tokamark] loading {split_name}: {len(shots)} shots "
              f"(batch_size={batch_size})", flush=True)
        for start in range(0, len(shots), batch_size):
            batch = shots[start:start + batch_size]
            ds = initialize_MAST_dataset(
                config_task=mapping["config"],
                shots_list=list(batch),
                local_flag=local,
                use_std_scaling=use_std_scaling,
                use_nan_filling=True,
                remove_outliers=True,
                remove_bad_efit_rating=True,
                store_manager_settings=store_manager_settings,
                verbose=False,
            )
            for i in range(len(ds)):
                shot_id = ds.get_shot_id(i)
                try:
                    shot = ds[i]
                except Exception as exc:
                    skipped += 1
                    if load_errors_shown < 5:
                        print(f"[tokamark] skip shot {shot_id}: load error {exc}")
                        load_errors_shown += 1
                    elif load_errors_shown == 5:
                        print("[tokamark] (further load errors suppressed)")
                        load_errors_shown += 1
                    continue
                pair = shot_to_yu(shot, mapping["y_names"], mapping["u_names"], dt)
                if pair is None:
                    del shot
                    skipped += 1
                    continue
                y, u = pair
                if feat_y_names is None:
                    feat_y_names, feat_u_names = expand_feature_names(mapping, shot)
                del shot
                if y.shape[1] != len(feat_y_names) or u.shape[1] != len(feat_u_names):
                    skipped += 1
                    continue
                n_windows += writer.add_shot(split_name, y, u, str(shot_id))
                del y, u
            del ds
            gc.collect()
            done = min(start + batch_size, len(shots))
            print(
                f"[tokamark] {split_name} progress {done}/{len(shots)} "
                f"shots_kept={writer.kept_shots} windows={n_windows} "
                f"skipped={skipped} pending_buf={writer._pending()}",
                flush=True,
            )

    if feat_y_names is None or feat_u_names is None or writer.kept_shots == 0:
        raise RuntimeError("no usable shots exported; check local zarr / S3 access")

    notes = (
        f"v1 map: y=output, u=actuator; dropped_inputs={mapping['dropped_inputs']}; "
        f"tokamark_task={mapping['task']}; official shot splits; streaming export"
    )
    writer.finalize(
        out_path,
        y_names=feat_y_names,
        u_names=feat_u_names,
        dataset=f"tokamark_{mapping['task']}",
        source="TokaMark / MAST",
        notes=notes,
    )
    if not keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)
    print(
        f"[tokamark] done kept_shots={writer.kept_shots} skipped={skipped} "
        f"d_y={len(feat_y_names)} d_u={len(feat_u_names)} dt={dt} "
        f"L={L} H={H} stride={stride}"
    )


def default_LH(mapping: dict[str, Any]) -> tuple[int, int, int]:
    """Choose L/H/stride from task window lengths and reference dt."""
    dt = _reference_dt(mapping)
    out_s = float(mapping.get("output_length_s") or 0.05)
    in_s = float(mapping.get("input_length_s") or out_s)
    span = max(out_s, in_s, 0.05)
    n = max(8, int(round(span / dt)))
    stride = max(1, n // 4)
    return n, n, stride


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="e.g. task_3-1 or task_2-1")
    ap.add_argument("--out", default=None, help="Output NPZ path (required unless --map-only)")
    ap.add_argument("--map-only", action="store_true",
                    help="Print y/u mapping and exit without loading shots")
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--local", action=argparse.BooleanOptionalAction, default=True,
                    help="Use local MAST zarr (default true)")
    ap.add_argument("--zarr-root", default=None,
                    help="Local MAST zarr root (overrides default /mast/tokamark/v1). "
                         "Example: /path/to/tokamark_data/data")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Shots loaded at once (default 4; use 1 if still killed)")
    ap.add_argument("--flush-windows", type=int, default=256,
                    help="Flush window buffer to a disk shard after this many windows")
    ap.add_argument("--keep-shards", action="store_true",
                    help="Keep temporary shard dir next to --out")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--use-std-scaling", action="store_true",
                    help="Keep TokaMark std scaling (default off; ICB normalizes)")
    args = ap.parse_args()

    _require_tokamark()
    mapping = map_task_channels(args.task)

    print(json.dumps({
        "task": mapping["task"],
        "y_names": mapping["y_names"],
        "u_names": mapping["u_names"],
        "dropped_inputs": mapping["dropped_inputs"],
        "input_length_s": mapping["input_length_s"],
        "output_length_s": mapping["output_length_s"],
        "stride_in_secs": mapping["stride_in_secs"],
    }, indent=2, ensure_ascii=False))

    if args.map_only:
        return

    if not args.out:
        raise SystemExit("--out is required unless --map-only")

    L, H, stride = default_LH(mapping)
    if args.L is not None:
        L = args.L
    if args.H is not None:
        H = args.H
    if args.stride is not None:
        stride = args.stride

    export_streaming(
        mapping,
        out_path=Path(args.out),
        L=L,
        H=H,
        stride=stride,
        local=args.local,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        use_std_scaling=args.use_std_scaling,
        zarr_root=args.zarr_root,
        batch_size=args.batch_size,
        flush_windows=args.flush_windows,
        keep_shards=args.keep_shards,
    )


if __name__ == "__main__":
    main()
