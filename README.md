# Industrial Forecasting Benchmark with Known Future Inputs

Industrial forecasters often receive an input trajectory specified before
prediction, such as a pump-voltage schedule or a sequence of valve commands. A
model of the resulting system response should preserve prefix consistency:
changing a later action must not change predictions before that action. This
repository provides the data, configurations, and code for evaluating that
setting on seven industrial datasets (NPPAD, Ventilator, PMSM, Cascaded Tanks,
Wiener–Hammerstein, Silverbox, and TokaMark Task 2–1) with eleven forecasting
models (iMamba, LSTM, KITE, DAG, CrossLinear, TiDE, TimeXer, TimeMixer,
PatchTST, DLinear, Persistence). Each dataset supplies historical targets,
historical exogenous inputs, and a known future control path; every
dataset–model entry is recorded in `configs/main_table_runs.json` with its
model configuration, training hyperparameters, and reported value.

## Getting started

```bash
pip install -r requirements.txt
```

Train an entry with its recorded configuration:

```bash
python train_main_table.py --dataset sysid_cascadedtanks --model imamba --device cuda
```

Evaluate the models against the reported table:

```bash
python eval_main_table.py --device cuda
```

Both scripts accept `--only <dataset>/<model>` or `--dataset`/`--model` to select
a single entry, and `--epochs 1` gives a quick smoke test.

## Repository layout

```
configs/main_table_runs.json   one record per dataset–model pair
data/                          windowed NPZ datasets
icb/                           models, adapters, evaluation protocol, training entry
third_party_repos/DAG/         vendored TSLib code used by six adapters
reformer_pytorch.py            vendored dependency of the TSLib attention layers
train_main_table.py            trains an entry with its recorded configuration
eval_main_table.py             evaluates trained models against the table
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
