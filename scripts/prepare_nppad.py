# -*- coding: utf-8 -*-
"""NPPAD(Nature Sci Data 2022, thu-inet/NuclearPowerPlantAccidentData)
→ 受控预测窗口 npz。

数据事实(已在真实数据上核对):
  * 每个场景一个 {severity}.mdb(Access),表 PlotData 为 97 列宽表,
    TIME 列不在第一列,必须按表头定位并按时间排序;约 497 步,dt≈10s。
  * 每个场景一份 {severity}Transient Report.txt,记录带时间戳的系统/
    操作员动作(Scram、泵/阀 Position Change、Malfunction 注入)。
    这些解析成阶跃函数即为控制信号 u(t)。
  * 依赖 mdbtools(Linux: apt install mdbtools;或用仓库自带的
    Windows pyodbc 脚本先转 csv,再用 --csv_root 指向)。

角色划分:
  * 内生 y:PlotData 全部连续状态量(默认剔除全程恒定列)。
  * 外生 u:报告解析的动作通道(逐组件保持最近指令值)
           + Malfunction 通道(事故注入强度,已知的外生扰动)
           + 可选 severity 静态协变量。
  * 初始位置启发式:名字含 Valve/FWIV/MSIV 的通道初始 1.0(常开),
    其余(Pump/Spray/PORV/Scram/Malfunction)初始 0.0。可用
    --init_override 'name=value' 覆盖。论文中如实说明该启发式。

切分在场景级(按 severity 洗牌)而非窗口级,避免泄漏。

用法:
  python make_nppad.py --root NPPAD --types SGBTR SGATR \
      --L 60 --H 60 --stride 30 --out nppad_sgtr.npz
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys

import numpy as np

RE_MALF = re.compile(r'([\d.]+)\s*sec,\s*Malfunction\s*#\s*(\d+)\s*Fraction\s*=\s*([\d.]+)')
RE_POS = re.compile(r'([\d.]+)\s*sec,\s*(.+?)\s*Position Change:\s*([\d.]+)\s*%')
RE_SCRAM = re.compile(r'([\d.]+)\s*sec,\s*Reactor Scram\s*$')


def read_plotdata_mdb(mdb_path):
    out = subprocess.run(['mdb-export', mdb_path, 'PlotData'],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f'mdb-export failed on {mdb_path}: {out.stderr[:200]}')
    rows = list(csv.reader(io.StringIO(out.stdout)))
    header, data = rows[0], rows[1:]
    arr = np.array(data, dtype=np.float64)
    t_idx = header.index('TIME')
    order = np.argsort(arr[:, t_idx], kind='stable')
    arr = arr[order]
    time = arr[:, t_idx]
    keep = [i for i in range(len(header)) if i != t_idx]
    names = [header[i] for i in keep]
    return time, arr[:, keep], names


def read_plotdata_csv(csv_path):
    rows = list(csv.reader(open(csv_path)))
    header, data = rows[0], rows[1:]
    arr = np.array(data, dtype=np.float64)
    t_idx = header.index('TIME')
    order = np.argsort(arr[:, t_idx], kind='stable')
    arr = arr[order]
    keep = [i for i in range(len(header)) if i != t_idx]
    return arr[:, t_idx], arr[:, keep], [header[i] for i in keep]


def parse_report(txt_path):
    """→ list[(t, channel_name, value_in_[0,1])],通道名规范化。"""
    events = []
    for line in open(txt_path, errors='ignore'):
        m = RE_MALF.search(line)
        if m:
            events.append((float(m.group(1)), f'Malfunction_{m.group(2)}',
                           float(m.group(3)) / 100.0))
            continue
        m = RE_POS.search(line)
        if m:
            name = re.sub(r'\s+', '_', m.group(2).strip())
            events.append((float(m.group(1)), name, float(m.group(3)) / 100.0))
            continue
        m = RE_SCRAM.search(line)
        if m:
            events.append((float(m.group(1)), 'Reactor_Scram', 1.0))
    return events


def default_init(channel):
    return 1.0 if re.search(r'valve|fwiv|msiv', channel, re.I) else 0.0


def events_to_grid(events, time_grid, vocab, init_override):
    """把事件序列栅格化为 [T, n_u] 的最近指令值(zero-order hold)。"""
    T = len(time_grid)
    u = np.zeros((T, len(vocab)), dtype=np.float64)
    for j, ch in enumerate(vocab):
        val = init_override.get(ch, default_init(ch))
        ev = sorted([(t, v) for (t, c, v) in events if c == ch])
        k = 0
        for i, tg in enumerate(time_grid):
            while k < len(ev) and ev[k][0] <= tg:
                val = ev[k][1]
                k += 1
            u[i, j] = val
    return u


def make_windows(y, u, L, H, stride):
    T = y.shape[0]
    out = []
    s = 0
    while s + L + H <= T:
        out.append((s, y[s:s + L], u[s:s + L],
                    u[s + L:s + L + H], y[s + L:s + L + H]))
        s += stride
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='NPPAD', help='含各事故类型子目录的根目录')
    ap.add_argument('--csv_root', default=None,
                    help='若已用仓库脚本转好 csv,指向其根目录(结构同 root)')
    ap.add_argument('--types', nargs='+', default=['SGBTR', 'SGATR'])
    ap.add_argument('--L', type=int, default=60)
    ap.add_argument('--H', type=int, default=60)
    ap.add_argument('--stride', type=int, default=30)
    ap.add_argument('--max_scenarios', type=int, default=0, help='每类型上限,0=全部')
    ap.add_argument('--drop_constant', action='store_true', default=True)
    ap.add_argument('--severity_covariate', action='store_true',
                    help='把 severity/100 作为常值外生通道加入 u')
    ap.add_argument('--init_override', nargs='*', default=[],
                    help="形如 Ctmt_Vent_Valve_#1=1.0 的初始值覆盖")
    ap.add_argument('--split', nargs=3, type=float, default=[0.7, 0.15, 0.15])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='nppad_sgtr.npz')
    args = ap.parse_args()

    init_override = {}
    for kv in args.init_override:
        k, v = kv.split('=')
        init_override[k] = float(v)

    # ---- 第一遍:收集场景清单与控制通道全集(词表跨场景一致) ----
    scenarios = []          # (type, severity, mdb_or_csv_path, report_path)
    vocab = set()
    for typ in args.types:
        d = os.path.join(args.root, typ)
        data_dir = os.path.join(args.csv_root, typ) if args.csv_root else d
        pattern = r'^(\d+)\.csv$' if args.csv_root else r'^(\d+)\.mdb$'
        sevs = sorted(
            {re.match(pattern, f).group(1)
             for f in os.listdir(data_dir) if re.match(pattern, f)},
            key=int,
        )
        if args.max_scenarios:
            sevs = sevs[:args.max_scenarios]
        for s in sevs:
            rep = os.path.join(d, f'{s}Transient Report.txt')
            if args.csv_root:
                dat = os.path.join(args.csv_root, typ, f'{s}.csv')
            else:
                dat = os.path.join(d, f'{s}.mdb')
            if not (os.path.exists(rep) and os.path.exists(dat)):
                continue
            scenarios.append((typ, int(s), dat, rep))
            for (_, ch, _) in parse_report(rep):
                vocab.add(ch)
    vocab = sorted(vocab)
    print(f'[nppad] {len(scenarios)} scenarios, {len(vocab)} control channels: {vocab}')

    # ---- 第二遍:逐场景栅格化并切窗 ----
    per_scn = []
    y_names = None
    for typ, sev, dat, rep in scenarios:
        time, Y, names = (read_plotdata_csv(dat) if args.csv_root
                          else read_plotdata_mdb(dat))
        if y_names is None:
            y_names = names
        elif names != y_names:
            raise RuntimeError(f'列不一致: {dat}')
        U = events_to_grid(parse_report(rep), time, vocab, init_override)
        if args.severity_covariate:
            U = np.concatenate([U, np.full((len(time), 1), sev / 100.0)], 1)
        per_scn.append((typ, sev, Y, U))

    if args.drop_constant:
        stacked = np.concatenate([Y for _, _, Y, _ in per_scn], 0)
        keep = np.where(stacked.std(0) > 1e-8)[0]
        y_names = [y_names[i] for i in keep]
        per_scn = [(t, s, Y[:, keep], U) for t, s, Y, U in per_scn]
        print(f'[nppad] kept {len(keep)} non-constant endogenous channels')

    # ---- 场景级切分 ----
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(per_scn))
    n_tr = int(args.split[0] * len(per_scn))
    n_va = int(args.split[1] * len(per_scn))
    part = {}
    part.update({i: 'train' for i in idx[:n_tr]})
    part.update({i: 'val' for i in idx[n_tr:n_tr + n_va]})
    part.update({i: 'test' for i in idx[n_tr + n_va:]})

    buf = {k: {q: [] for q in ('y_hist', 'u_hist', 'u_future', 'y_future')}
           for k in ('train', 'val', 'test')}
    scn_of = {k: [] for k in ('train', 'val', 'test')}
    start_of = {k: [] for k in ('train', 'val', 'test')}
    for i, (typ, sev, Y, U) in enumerate(per_scn):
        for (start, yh, uh, uf, yf) in make_windows(Y, U, args.L, args.H, args.stride):
            k = part[i]
            buf[k]['y_hist'].append(yh); buf[k]['u_hist'].append(uh)
            buf[k]['u_future'].append(uf); buf[k]['y_future'].append(yf)
            scn_of[k].append(f'{typ}_{sev}')
            start_of[k].append(start)

    save = {}
    for k in buf:
        for q in buf[k]:
            save[f'{k}_{q}'] = np.stack(buf[k][q]).astype(np.float32) \
                if buf[k][q] else np.zeros((0,))
        save[f'{k}_scenario_id'] = np.asarray(scn_of[k], dtype=np.str_)
        save[f'{k}_window_start'] = np.asarray(start_of[k], dtype=np.int64)
    ytr = save['train_y_hist']
    mu = ytr.reshape(-1, ytr.shape[-1]).mean(0)
    sd = ytr.reshape(-1, ytr.shape[-1]).std(0) + 1e-8
    utr = save['train_u_hist']
    umu = utr.reshape(-1, utr.shape[-1]).mean(0)
    usd = utr.reshape(-1, utr.shape[-1]).std(0) + 1e-8
    meta = dict(y_names=y_names,
                u_names=vocab + (['severity'] if args.severity_covariate else []),
                L=args.L, H=args.H, stride=args.stride, types=args.types,
                n_scenarios=len(per_scn),
                windows={k: len(scn_of[k]) for k in scn_of},
                roles={
                    'y': {name: 'measured_target' for name in y_names},
                    'u': {name: 'future_control_or_disturbance' for name in vocab},
                    **({'severity': 'static_scenario_covariate'}
                       if args.severity_covariate else {}),
                },
                future_availability=(
                    'offline supplied accident/action trajectory; verify online '
                    'availability before calling it a deployable control'
                ))
    np.savez_compressed(args.out, **save, y_mean=mu, y_std=sd,
                        u_mean=umu, u_std=usd,
                        meta=json.dumps(meta, ensure_ascii=False))
    print(f'[nppad] saved {args.out} | windows:', meta['windows'],
          f'| d_y={len(y_names)}, d_u={len(meta["u_names"])}')


if __name__ == '__main__':
    main()
