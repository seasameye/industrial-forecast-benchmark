# Industrial Forecasting Benchmark with Known Future Inputs

Industrial forecasting tasks often provide the input trajectory before
prediction — a pump-voltage schedule, a sequence of valve commands. A model of
the resulting response should preserve **prefix consistency**: changing a later
action must not change predictions before that action.

This repository provides seven industrial datasets and ten forecasting models
under this setting. Each dataset supplies historical targets, historical
exogenous inputs, and the future trajectory of its **control variables** — the
actuator commands and setpoints that are fixed before prediction (for example a
pump-voltage schedule or a sequence of valve commands) and that the model must
use as a known input over the horizon.

## Getting started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train a model on a dataset:

```bash
python train_main_table.py --dataset sysid_cascadedtanks --model lstm --device cuda
```

Evaluate a trained model:

```bash
python eval_main_table.py --device cuda
```

Select a single entry with `--only <dataset>/<model>` (evaluation) or
`--dataset` / `--model` (training). Add `--epochs 1` for a quick smoke test.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | windowed NPZ datasets |
| `configs/main_table_runs.json` | configurations for each dataset–model pair |
| `icb/` | models, adapters, evaluation protocol, training entry |
| `third_party_repos/DAG/` | vendored TSLib code used by six adapters |
| `reformer_pytorch.py` | vendored dependency of the TSLib attention layers |
| `train_main_table.py` | trains a model on a dataset |
| `eval_main_table.py` | evaluates a trained model |

## Note

TokaMark Task 2–1 is not included because its windowed dataset is 7.2 GB. The
raw measurements are publicly available from the original TokaMark publication
and can be windowed into the same format as the other datasets.
