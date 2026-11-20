# Main-table release: industrial forecasting with known future inputs
Anonymous code, data, and trained checkpoints for review.

## Contents
- `data/` windowed NPZ datasets (train/val/test histories and future controls,
  normalization statistics, and window metadata). Window counts match Table I.
- `checkpoints/<dataset>/<model>.pt` checkpoints for iMamba, LSTM, DLinear, and
  Persistence (the fast path), each with `<model>.json` recording the model
  configuration, the training configuration, the source run, and the reproduced
  metrics. The configuration JSONs for every entry are included regardless.
- `checkpoints_full.zip` (release asset) the remaining seven model families
  (KITE, DAG, CrossLinear, TimeXer, TiDE, TimeMixer, PatchTST). Unzip it into the
  repository root to enable all 66 entries.
- `table2.json` the reported main-table values (normalized MSE) per entry.
- `icb/` models, adapters, protocol, and the training entry point.
- `third_party_repos/DAG/` and `reformer_pytorch.py` vendored dependencies of the
  TSLib-based adapters.
- `eval_main_table.py` evaluates every released checkpoint and compares the
  result with `table2.json`.

## Reproduce the table (no training)
```bash
pip install -r requirements.txt
python eval_main_table.py --device cuda          # iMamba / LSTM / DLinear / Persistence
unzip checkpoints_full.zip                       # optional, adds the other seven
python eval_main_table.py --device cuda          # all 66 entries
```
Each line prints the reproduced normalized MSE next to the reported value;
entries whose checkpoint is not present are reported as SKIP.

## Retrain one entry (optional)
```bash
python -m icb.train --data data/sysid_cascadedtanks.npz --model imamba \
    --model_config checkpoints/sysid_cascadedtanks/imamba.json \
    --epochs 40 --bs 64 --lr 1e-3 --seed 0 --device cuda --out_dir runs
```
(The `--model_config` flag expects a JSON file whose contents are the
`model_config` object; the checkpoint records it in that form.)

## Notes
- All entries are a single run with seed 0.
- Deterministic checkpoints reproduce the reported value to within 0.2%;
  KITE draws eight Monte-Carlo samples at inference time and reproduces the
  reported value to within 1%.
- PMSM/LSTM: the table prints 0.00036; the released checkpoint reproduces
  0.000355, i.e. the printed value is a rounding of the same run.
- Cascaded Tanks uses the split of Table I (60/19/91 windows).
- TokaMark is omitted from this release.
