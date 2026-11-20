# Industrial Forecasting Benchmark with Known Future Inputs

Industrial forecasting tasks often provide the input trajectory before
prediction — a pump-voltage schedule, a sequence of valve commands. A model of
the resulting response should preserve **prefix consistency**: changing a later
action must not change predictions before that action.

This repository contains the data, configurations, and code for evaluating that
setting: seven industrial datasets × eleven forecasting models, with every
dataset–model entry recorded in `configs/main_table_runs.json`.

## Getting started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train an entry with its recorded configuration:

```bash
python train_main_table.py --dataset sysid_cascadedtanks --model imamba --device cuda
```

Evaluate against the reported table:

```bash
python eval_main_table.py --device cuda
```

Select a single entry with `--only <dataset>/<model>` (evaluation) or
`--dataset` / `--model` (training). Add `--epochs 1` for a quick smoke test.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | windowed NPZ datasets |
| `configs/main_table_runs.json` | configuration, hyperparameters, and reported value of every entry |
| `icb/` | models, adapters, evaluation protocol, training entry |
| `third_party_repos/DAG/` | vendored TSLib code used by six adapters |
| `reformer_pytorch.py` | vendored dependency of the TSLib attention layers |
| `train_main_table.py` | trains an entry with its recorded configuration |
| `eval_main_table.py` | evaluates trained models against the table |

## TokaMark

TokaMark Task 2–1 is not included: its windowed dataset is 7.2 GB. It is
available on request, or can be rebuilt from the original measurements with
L = 20, H = 100 and 381068 / 44872 / 47363 train/val/test windows, using the
same NPZ keys as the other datasets.
