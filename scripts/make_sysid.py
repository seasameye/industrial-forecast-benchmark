# -*- coding: utf-8 -*-
"""make_sysid.py — 非线性系统辨识基准(nonlinearbenchmark.org)→ 统一窗口 npz。

依赖:pip install nonlinear_benchmarks(首次运行会自动下载数据)
支持:CascadedTanks | WienerHammer | Silverbox | CED(可扩展)

这些基准就是显式的 (u, y) 受控系统,任务即"已知未来 u 的自由仿真"
(free-run simulation)——与我们的受控预测设定一字不差。基准自带
train/test 划分:train_val 段按时间切 train/val(避免泄漏),
官方 test 段(可能是多段)全部切窗作为 test。

用法:python make_sysid.py --bench CascadedTanks --L 60 --H 60 --out ct.npz
"""

import argparse
import json

import numpy as np

BENCH = {
    'CascadedTanks': 'Cascaded_Tanks',
    'WienerHammer': 'WienerHammerBenchMark',
    'Silverbox': 'Silverbox',
    'CED': 'CED',
    'EMPS': 'EMPS',
}


def to_seq(obj):
    """nonlinear_benchmarks 的返回可能是单段或段元组,统一为 [(u,y)] 列表。"""
    if isinstance(obj, (list, tuple)):
        out = []
        for o in obj:
            out.extend(to_seq(o))
        return out
    u = np.atleast_2d(np.asarray(obj.u, dtype=np.float32))
    y = np.atleast_2d(np.asarray(obj.y, dtype=np.float32))
    if u.shape[0] < u.shape[1]:
        u = u.T
    if y.shape[0] < y.shape[1]:
        y = y.T
    return [(u, y)]


def windows(u, y, L, H, stride):
    T = len(y)
    out = []
    s = 0
    while s + L + H <= T:
        out.append((y[s:s + L], u[s:s + L], u[s + L:s + L + H],
                    y[s + L:s + L + H]))
        s += stride
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True, choices=sorted(BENCH))
    ap.add_argument('--L', type=int, default=60)
    ap.add_argument('--H', type=int, default=60)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    try:
        import nonlinear_benchmarks as nlb
    except ImportError:
        raise SystemExit('请先 pip install nonlinear_benchmarks')
    fn = getattr(nlb, BENCH[args.bench])
    train_val, test = fn()

    buf = {k: {q: [] for q in ('y_hist', 'u_hist', 'u_future', 'y_future')}
           for k in ('train', 'val', 'test')}

    for u, y in to_seq(train_val):
        cut = int(len(y) * (1 - args.val_frac))
        for part, (uu, yy) in (('train', (u[:cut], y[:cut])),
                               ('val', (u[cut:], y[cut:]))):
            for (yh, uh, uf, yf) in windows(uu, yy, args.L, args.H, args.stride):
                buf[part]['y_hist'].append(yh); buf[part]['u_hist'].append(uh)
                buf[part]['u_future'].append(uf); buf[part]['y_future'].append(yf)
    for u, y in to_seq(test):
        for (yh, uh, uf, yf) in windows(u, y, args.L, args.H, args.stride):
            buf['test']['y_hist'].append(yh); buf['test']['u_hist'].append(uh)
            buf['test']['u_future'].append(uf); buf['test']['y_future'].append(yf)

    save = {}
    for k in buf:
        for q in buf[k]:
            save[f'{k}_{q}'] = np.stack(buf[k][q]).astype(np.float32)
    d_y = save['train_y_hist'].shape[-1]
    d_u = save['train_u_hist'].shape[-1]
    ytr = save['train_y_hist'].reshape(-1, d_y)
    utr = save['train_u_hist'].reshape(-1, d_u)
    meta = dict(bench=args.bench, L=args.L, H=args.H, stride=args.stride,
                windows={k: len(buf[k]['y_hist']) for k in buf})
    out = args.out or f'sysid_{args.bench.lower()}.npz'
    np.savez_compressed(out, **save,
                        y_mean=ytr.mean(0), y_std=ytr.std(0) + 1e-8,
                        u_mean=utr.mean(0), u_std=utr.std(0) + 1e-8,
                        meta=json.dumps(meta))
    print(f'[sysid:{args.bench}] saved {out} | windows:', meta['windows'],
          f'| d_y={d_y} d_u={d_u}')


if __name__ == '__main__':
    main()
