# Data sources and preprocessing

`data/*.npz` are windowed versions of six public industrial datasets. Raw data
are not redistributed here: each file can be rebuilt from its public source with
the script and command below.

## Sources

| Dataset | File in `data/` | Source | License | Script |
|---|---|---|---|---|
| NPPAD | `nppad_sgtr_l60_h60.npz` | [thu-inet/NuclearPowerPlantAccidentData](https://github.com/thu-inet/NuclearPowerPlantAccidentData) (PCTRAN accident scenarios) | MIT | `scripts/prepare_nppad.py` |
| Ventilator | `ventilator_l40_h40.npz` | [Google Brain Ventilator Pressure Prediction](https://www.kaggle.com/competitions/ventilator-pressure-prediction) (Kaggle, `train.csv`) | Kaggle competition terms | `scripts/prepare_ventilator.py` |
| PMSM | `pmsm_l60_h60.npz` | [Electric Motor Temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) (`measures_v2.csv`, Paderborn University) | CC BY-SA 4.0 | `scripts/prepare_pmsm.py` |
| Cascaded Tanks | `sysid_cascadedtanks.npz` | [nonlinear_benchmarks](https://doi.org/10.5281/zenodo.20819449) (Schoukens) | CC BY 4.0 | `scripts/make_sysid.py` |
| Wiener–Hammerstein | `sysid_wienerhammer.npz` | same | CC BY 4.0 | `scripts/make_sysid.py` |
| Silverbox | `sysid_silverbox.npz` | same | CC BY 4.0 | `scripts/make_sysid.py` |

The raw files of the three nonlinear benchmarks are downloaded automatically by
the `nonlinear_benchmarks` package; NPPAD, Ventilator, and PMSM expect a local
copy of the respective raw file.

## Rebuilding

Install the extra dependencies first:

```bash
pip install pandas scipy nonlinear_benchmarks
```

```bash
# NPPAD (root = the downloaded repository)
python scripts/prepare_nppad.py --root /path/to/NuclearPowerPlantAccidentData \
    --types SGBTR SGATR --L 60 --H 60 --stride 30 --out data/nppad_sgtr_l60_h60.npz

# Ventilator (train.csv from the Kaggle competition)
python scripts/prepare_ventilator.py --train_csv /path/to/train.csv \
    --L 40 --H 40 --stride 40 --out data/ventilator_l40_h40.npz

# PMSM (measures_v2.csv from the Kaggle dataset)
python scripts/prepare_pmsm.py --csv /path/to/measures_v2.csv \
    --L 60 --H 60 --stride 60 --out data/pmsm_l60_h60.npz

# Nonlinear benchmarks (data downloaded by the package)
# Cascaded Tanks uses a 0.30 validation fraction: at 0.15 the validation
# windows fall on a short low-amplitude segment.
python scripts/make_sysid.py --bench CascadedTanks --L 60 --H 60 --stride 10 \
    --val_frac 0.30 --out data/sysid_cascadedtanks.npz
python scripts/make_sysid.py --bench WienerHammer --L 60 --H 60 --stride 10 \
    --out data/sysid_wienerhammer.npz
python scripts/make_sysid.py --bench Silverbox --L 60 --H 60 --stride 10 \
    --out data/sysid_silverbox.npz
```

## Format

Every file stores the same keys:

| Key | Shape | Meaning |
|---|---|---|
| `train_y_hist`, `val_y_hist`, `test_y_hist` | (N, L, d_y) | past targets |
| `train_u_hist`, … | (N, L, d_u) | past control inputs |
| `train_u_future`, … | (N, H, d_u) | known future control inputs |
| `train_y_future`, … | (N, H, d_y) | forecast targets |
| `y_mean`, `y_std`, `u_mean`, `u_std` | (d, ) | normalization statistics, computed on the training split only |
| `meta` | JSON string | names, window parameters, split sizes |

Windows are cut inside each split, so no window crosses a split boundary.

Running the three `make_sysid.py` commands and the `prepare_pmsm.py` command
above reproduces the shipped `sysid_*.npz` and `pmsm_l60_h60.npz` files exactly
(identical arrays, normalization statistics, and split sizes).
Inputs and targets are stored in physical units; normalization is applied at
training and evaluation time using the stored statistics.

## TokaMark

TokaMark Task 2–1 is not included because the windowed file is 7.2 GB.

| Property | Value |
|---|---|
| Task | `task_2-1` |
| Targets $\mathbf{y}$ | 29 plasma-current and equilibrium quantities |
| Future controls $\mathbf{u}$ | 4 coil voltages + neutral-beam injection power |
| Window | $L=20$, $H=100$, stride $=40$ |
| Split | 381068 / 44872 / 47363 (official random shot split, fixed discharge-ID lists, only shots containing all required signals) |

The raw shots come from the TokaMark release. Once the `tokamark` package and
its MAST zarr store are available:

```bash
python scripts/prepare_tokamark_task.py --task task_2-1 \
    --zarr-root /path/to/tokamark_data/data \
    --L 20 --H 100 --stride 40 \
    --out data/tokamark_task_2-1.npz
```

The script streams shot batches, windows each split independently, and writes
the same NPZ keys as the other datasets (it requires Python 3.9 or newer). `--map-only` prints the channel
mapping without loading any shots, and `--max-train/--max-val/--max-test` build
a smaller subset for smoke tests.

Because $H=100>L=20$, the four channel-concatenating adapters (CrossLinear,
TimeMixer, PatchTST, DLinear) average-pool the future path from $H$ to $L$
bins before concatenation; the other models consume the full 100 steps through
their native interfaces.
