# Industrial Forecasting Benchmark with Known Future Inputs

This repository contains the data, configurations, and code behind the main table
of the paper. It covers seven industrial forecasting tasks (NPPAD, Ventilator,
PMSM, Cascaded Tanks, Wiener–Hammerstein, Silverbox, and TokaMark Task 2–1),
each with historical targets, historical exogenous inputs, and a known future
control path, together with eleven forecasting models (iMamba, LSTM, KITE, DAG,
CrossLinear, TiDE, TimeXer, TimeMixer, PatchTST, DLinear, Persistence).

Every "dataset × model" entry is stored in `configs/main_table_runs.json` with its
model configuration, training hyperparameters, and reported value.

## Getting started

```bash
pip install -r requirements.txt
```

Trained weights are distributed as a release asset (`checkpoints_all.zip`).
Unzip it into the repository root to obtain `checkpoints/<dataset>/<model>.pt`,
then reproduce the table:

```bash
python eval_main_table.py --device cuda
```

Alternatively, train an entry with the recorded configuration:

```bash
python train_main_table.py --dataset sysid_cascadedtanks --model imamba --device cuda
```

Both scripts accept `--only <dataset>/<model>` (evaluation) or `--dataset`/`--model`
(training) to select a single entry; `--epochs 1` gives a quick smoke test.

## Repository layout

```
configs/main_table_runs.json   one record per dataset–model pair
data/                          windowed NPZ datasets
icb/                           models, adapters, evaluation protocol, training entry
third_party_repos/DAG/         vendored TSLib code used by six adapters
reformer_pytorch.py            vendored dependency of the TSLib attention layers
train_main_table.py            trains an entry with its recorded configuration
eval_main_table.py             evaluates released checkpoints against the table
```

## TokaMark data

The TokaMark Task 2–1 windowed dataset (29 targets, 5 future controls, L = 20,
H = 100, 381068 / 44872 / 47363 train/val/test windows) is **not** included
because it is 7.2 GB, which exceeds the GitHub limits. It is available on request
from the authors, and can be rebuilt from the original Task 2–1 measurements using
the splits above and the same NPZ keys as the other datasets
(`train_y_hist`, `train_u_hist`, `train_u_future`, `train_y_future`, plus the
`*_mean` / `*_std` statistics). Once placed at `data/tokamark_task_2-1.npz`,
`train_main_table.py --dataset tokamark_task_2-1 --model imamba` works as for any
other dataset; on this task iMamba uses d = 96, N = 16, dropout 0.05, and
iMamba/LSTM use batch size 256.
