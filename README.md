# 工业时序预测基准：已知未来输入下的滚动预测

本仓库提供论文主表（7 个数据集 × 11 个模型）的**窗口数据、配置与代码**，
以及复现主表所需的训练与评测脚本。

## 1. 仓库内容

| 路径 | 说明 |
|---|---|
| `data/` | 6 个已切好窗口的 NPZ 数据集（历史目标、历史外生输入、未来外生输入、归一化统计量） |
| `configs/main_table_runs.json` | **唯一配置来源**：77 条记录（6×11 + TokaMark 11），含模型配置、训练超参、表值 |
| `icb/` | 模型实现、评测协议（`protocol.py`）、训练入口（`train.py`） |
| `third_party_repos/DAG/`、`reformer_pytorch.py` | CrossLinear / TiDE / TimeXer / TimeMixer / PatchTST / DLinear 依赖的 TSLib 代码 |
| `train_main_table.py` | 按记录的配置训练主表条目 |
| `eval_main_table.py` | 加载已训练权重，复现主表数值 |

> 本仓库**不含 checkpoint**（体积原因）。训练权重通过论文发布页的
> `checkpoints_all.zip` 获取，或直接用 `train_main_table.py` 重新训练（见第 3 节）。

## 2. 环境

```bash
python -m venv .venv && source .venv/bin/activate     # 建议 Python >= 3.10
pip install -r requirements.txt
```

需要 PyTorch ≥ 2.0；有 GPU 时自动使用 CUDA，也可 `--device cpu`。

## 3. 复现主表

### 方式 A：用发布页提供的 checkpoint（最快）

把 `checkpoints_all.zip` 解压到仓库根目录，得到 `checkpoints/<数据集>/<模型>.pt`，然后：

```bash
python eval_main_table.py --device cuda
```

逐条打印 `复现值 (表值, 相对误差) OK/MISMATCH`，最后给出通过条数。
只跑一条：

```bash
python eval_main_table.py --device cuda --only sysid_cascadedtanks/imamba
```

### 方式 B：从头训练（无需 checkpoint）

```bash
# 训练一条，超参全部取自 configs/main_table_runs.json
python train_main_table.py --dataset sysid_cascadedtanks --model imamba --device cuda

# 冒烟检查（1 个 epoch，确认流程通）
python train_main_table.py --dataset sysid_cascadedtanks --model imamba --epochs 1 --device cuda

# 某个数据集的所有基线
python train_main_table.py --dataset sysid_cascadedtanks --all-models --device cuda
```

训练结果写入 `runs/<数据集>_<模型>_seed0.json`，脚本会自动打印训练值与表值的相对误差。

已实测（从头训练，40 epochs，seed 0）：

| 数据集 | 模型 | 训练结果 | 表值 |
|---|---|---|---|
| Cascaded Tanks | iMamba | 0.142498 | 0.1425 |
| Cascaded Tanks | LSTM | 0.191781 | 0.19178 |
| Cascaded Tanks | DLinear | 0.913203 | 0.9132 |
| Cascaded Tanks | Persistence | 1.050660 | 1.05066 |
| Wiener–Hammerstein | DLinear | 0.833127 | 0.83313 |

## 4. 数据集

| 数据集 | 文件 | 目标维数 $d_y$ | 输入维数 $d_u$ | L / H | train / val / test | 大小 |
|---|---|---|---|---|---|---|
| NPPAD | `nppad_sgtr_l60_h60.npz` | 64 | 10 | 60 / 60 | 1752 / 377 / 386 | 22 MB |
| Ventilator | `ventilator_l40_h40.npz` | 1 | 4 | 40 / 40 | 52815 / 11317 / 11318 | 28 MB |
| PMSM | `pmsm_l60_h60.npz` | 4 | 8 | 60 / 60 | 16008 / 2259 / 3812 | 100 MB |
| Cascaded Tanks | `sysid_cascadedtanks.npz` | 1 | 1 | 60 / 60 | 60 / 19 / 91 | 32 KB |
| Wiener–Hammerstein | `sysid_wienerhammer.npz` | 1 | 1 | 60 / 60 | 8489 / 1489 / 7869 | 2.5 MB |
| Silverbox | `sysid_silverbox.npz` | 1 | 1 | 60 / 60 | 5519 / 965 / 9382 | 2.5 MB |
| **TokaMark Task 2-1** | 未随仓库发布 | 29 | 5 | 20 / 100 | 381068 / 44872 / 47363 | **7.2 GB** |

以上划分与论文 Table I 完全一致（可用 `data/*.npz` 里的 `train_y_hist` 等数组核对）。

每个 NPZ 的键：`train_y_hist`、`train_u_hist`、`train_u_future`、`train_y_future`
（`val_*`、`test_*` 同理），以及归一化统计量 `y_mean`、`y_std`、`u_mean`、`u_std`。

### 为什么没有 TokaMark 数据

TokaMark Task 2-1 的窗口化 NPZ 有 **7.2 GB**（381k 训练窗口 × 20 步历史 × 29 个目标，
以及 100 步未来），超出 GitHub 单文件 100 MB 和仓库 1 GB 的限制，因此不放进仓库。

获取方式（二选一）：

1. **向作者索取**：按论文通信地址联系，我们会提供与上表完全一致的窗口化 NPZ；
2. **自行重建**：从原始 TokaMark Task 2-1 测量数据出发，按 ${L}=20$、${H}=100$、
   步长 1，划分 381068 / 44872 / 47363（train/val/test），
   并按第 4 节的键名与统计量格式保存为 `data/tokamark_task_2-1.npz`。

重建后即可用同一套脚本训练与评测：

```bash
python train_main_table.py --dataset tokamark_task_2-1 --model imamba --device cuda
```

TokaMark 的特殊设置：iMamba 用 $d=96$、$N=16$、dropout 0.05；iMamba 与 LSTM 的
batch size 为 256；因 $H=100 > L=20$，通用适配器（TimeMixer / PatchTST / DLinear）
只消费前 20 个未来控制步（见补充材料 S-II-B）。

## 5. 配置文件

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

## 6. 目录速查

```
configs/main_table_runs.json   77 条配置与表值
data/                          6 个 NPZ（TokaMark 另见第 4 节）
icb/models/                    各模型实现与适配器
icb/protocol.py                数据加载、归一化、评测指标
icb/train.py                   单条训练入口
train_main_table.py            按配置训练
eval_main_table.py             加载权重复现主表
```
