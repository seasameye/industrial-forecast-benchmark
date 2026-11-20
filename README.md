# 工业时序预测基准：已知未来输入下的滚动预测

本仓库是论文主表（7 个数据集 × 11 个模型）的**数据、配置与评测代码**。
所有数字都可以用这里的窗口数据 + 配置 + 代码复现。

包含内容：

- `data/`：已切好窗口的 NPZ 数据集（历史目标、历史外生输入、未来外生输入、
  归一化统计量、窗口元数据）。窗口数见下表的 train/val/test。
- `checkpoints/`：主表每个"数据集 × 模型"的已训练权重（`.pt`）。
- `configs/main_table_runs.json`：**唯一的配置来源**。每条记录包含数据集名、
  模型名、模型配置、训练超参、checkpoint 路径和对应的表值。
- `icb/`：模型实现、评测协议（`protocol.py`）、训练入口（`train.py`）。
- `third_party_repos/DAG/`、`reformer_pytorch.py`：CrossLinear / TiDE / TimeXer /
  TimeMixer / PatchTST / DLinear 依赖的 TSLib 代码。
- `eval_main_table.py`：加载 checkpoint 直接复现主表。
- `train_main_table.py`：按记录的配置从头训练主表条目。

## 1. 环境

```bash
python -m venv .venv && source .venv/bin/activate     # 建议 Python >= 3.10
pip install -r requirements.txt
```

需要 PyTorch ≥ 2.0；有 GPU 时自动用 CUDA，也可以 `--device cpu`。

## 2. 复现主表（不需要训练）

```bash
python eval_main_table.py --device cuda
```

逐条打印 `复现值 (表值, 相对误差) OK/MISMATCH`，最后给通过条数。
只想看某一条：

```bash
python eval_main_table.py --device cuda --only sysid_cascadedtanks/imamba
```

说明：

- KITE 推理含 8 次蒙特卡洛采样，复现误差 < 1%；其余模型 < 0.2%。
- `pmsm_l60_h60/lstm` 表中印 0.00036，checkpoint 实际 0.000355（四舍五入）。
- 仓库自带 iMamba / LSTM / DLinear / Persistence 四个模型的权重（每个数据集 4 个）；
  其余 7 个基线（KITE / DAG / CrossLinear / TimeXer / TiDE / TimeMixer / PatchTST）
  可以用第 3 节的训练脚本重训，或解压发布页的 `checkpoints_full.zip` 到 `checkpoints/`。
  缺少某个权重时，评测会打印 SKIP 并继续。

## 3. 训练复现（从头训练）

```bash
# 训练一条，超参全部取自 configs/main_table_runs.json
python train_main_table.py --dataset sysid_cascadedtanks --model imamba --device cuda

# 冒烟检查：只跑 1 个 epoch，确认流程通
python train_main_table.py --dataset sysid_cascadedtanks --model imamba --epochs 1 --device cuda

# 训练某个数据集的所有基线
python train_main_table.py --dataset sysid_cascadedtanks --all-models --device cuda
```

训练结果写到 `runs/<数据集>_<模型>_seed0.json`，脚本会自动打印
`trained mse_norm` 与表值的相对误差。

## 4. 数据集与配置

| 数据集 | 文件 | 目标维数 | 外生输入维数 | L / H | train / val / test |
|---|---|---|---|---|---|
| NPPAD | `data/nppad_sgtr_l60_h60.npz` | 64 | 10 | 60 / 60 | 1752 / 377 / 386 |
| Ventilator | `data/ventilator_l40_h40.npz` | 1 | 4 | 40 / 40 | 52815 / 11317 / 11318 |
| PMSM | `data/pmsm_l60_h60.npz` | 4 | 8 | 60 / 60 | 16008 / 2259 / 3812 |
| Cascaded Tanks | `data/sysid_cascadedtanks.npz` | 1 | 1 | 60 / 60 | 60 / 19 / 91 |
| Wiener–Hammerstein | `data/sysid_wienerhammer.npz` | 1 | 1 | 60 / 60 | 8489 / 1489 / 7869 |
| Silverbox | `data/sysid_silverbox.npz` | 1 | 1 | 60 / 60 | 5519 / 965 / 9382 |
| TokaMark Task 2-1 | 不随仓库发布 | 29 | 5 | 20 / 100 | 381068 / 44872 / 47363 |

`configs/main_table_runs.json` 的字段：

```json
{
  "dataset": "sysid_cascadedtanks",
  "model": "imamba",
  "model_config": {"d_model": 64, "n_state": 8, "n_layers": 2, "dropout": 0.0},
  "training_config": {"epochs": 40, "batch_size": 64, "learning_rate": 0.001,
                      "weight_decay": 0.0001, "patience": 8},
  "checkpoint": "checkpoints/sysid_cascadedtanks/imamba.pt",
  "table_entry": {"mse_normalized": 0.1425, "mae_normalized": 0.26749}
}
```

## 5. TokaMark

TokaMark 的窗口数据（7.2 GB）和 checkpoint 不在本仓库，但配置仍列在
`configs/main_table_runs.json` 中（`"checkpoint": null`），评测时会打印 SKIP。
按上面的数据格式重建 NPZ 后，可用 `train_main_table.py --dataset tokamark_task_2-1`
训练。该数据集上 iMamba 用 d=96、N=16、dropout 0.05，iMamba/LSTM 的 batch size 为 256；
因 H=100 > L=20，通用适配器只消费前 20 个未来控制步（见补充材料 S-II-B）。

## 6. 目录速查

```
data/                       窗口数据（6 个 NPZ）
configs/main_table_runs.json 77 条配置（6×11 + TokaMark 11）
checkpoints/<数据集>/*.pt    已训练权重
icb/models/                 各模型实现与适配器
icb/protocol.py             数据加载、归一化、评测指标
icb/train.py                单条训练入口
eval_main_table.py          复现主表
train_main_table.py         按配置训练
```
