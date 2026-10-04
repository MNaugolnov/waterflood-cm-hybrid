"""Проверка на полусинтетике: что можно восстановить при реальной закачке и реальном шуме.

Пример:
    python run_synth_check.py --data data --blocks S1

Берутся реальные закачка, график работы и координаты участка. Дебиты считаются по заданной
(«истинной») ёмкостной модели и зашумляются. Два режима:
    сильный — закачка даёт существенную часть добычи (сумма f по нагнетательной 0.3-0.9);
    слабый  — закачка почти не влияет (сумма f 0.03-0.10), остальная добыча от неё не зависит.
Для каждого режима сравниваются с истиной: событийная оценка отклика, CM и модель отклика.
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from hybrid.cm import CM
from hybrid.synth import make_truth, make_dataset
from hybrid.response import find_events, observed_response, model_response, WINDOWS
from hybrid.respmodel import ResponseModel
from run_hybrid import load_blocks
from run_response import per_neighbor

warnings.filterwarnings('ignore')
FWD = [w for w, (a, _) in WINDOWS.items() if a > 0]
W = '17-26 нед'


def run(name, gain, blocks, seed=0):
    tabs, items, info, store = [], [], [], {}
    for b, ds0 in blocks.items():
        sd = seed + sum(map(ord, b))
        truth = make_truth(ds0, total_gain=gain, seed=sd)
        ds = make_dataset(ds0, truth, seed=sd)
        cms, theta = {}, None
        for g in range(52, ds.T - 8, 13):
            cm = CM(ds.mask, ds.dist, step=ds.step).fit(ds.inj[:g], ds.up[:g], ds.liq[:g], theta0=theta,
                                                        maxiter=2000 if theta is None else 400)
            theta, cms[g] = cm.theta, cm
        f_dict = {g: cm.f for g, cm in cms.items()}
        items.append((ds, f_dict, 52, ds.T))
        ev = find_events(ds)
        ev = ev[ev.t >= 52]
        t = observed_response(ds, ev)
        tru = model_response(lambda inj, up: truth.simulate(inj, up)[1], ds, ev)
        t = t.merge(tru.rename(columns={'мод ' + w: 'истина ' + w for w in FWD}), on=['ev', 'j'])
        parts = [model_response(lambda inj, up, cm=cm: cm.simulate(inj, up)[1], ds, ev[(ev.t >= g) & (ev.t < g + 13)])
                 for g, cm in cms.items() if ((ev.t >= g) & (ev.t < g + 13)).any()]
        t = t.merge(pd.concat(parts).rename(columns={'мод ' + w: 'CM ' + w for w in FWD}), on=['ev', 'j'])
        t['block'] = b
        tabs.append(t)
        store[b] = (ds, ev, f_dict)
        last, m = cms[max(cms)], ds.mask
        info.append((b, spearmanr(last.f[m], truth.f[m]).correlation, last.f.sum(axis=1).mean(),
                     truth.f.sum(axis=1).mean(), np.median(last.tau), np.median(truth.tau)))
    rm = ResponseModel().fit(items)
    for k, (b, (ds, ev, f_dict)) in enumerate(store.items()):
        p = pd.concat([model_response(rm.simulator(ds, f_dict, e.t), ds, ev[ev.ev == e.ev]) for e in ev.itertuples()])
        tabs[k] = tabs[k].merge(p.rename(columns={'мод ' + w: 'R ' + w for w in FWD}), on=['ev', 'j'])
        tabs[k]['ev'] = b + '-' + tabs[k].ev.astype(str)
    t = pd.concat(tabs)
    n = t[t.dist <= 1000].dropna(subset=[W])
    s = np.sign(n.d)
    obs = per_neighbor(n, W)
    print(f'\n=== Режим «{name}»: сумма f по нагнетательной {gain[0]}-{gain[1]}; событий {obs[3]}')
    print(f'Отклик соседа на +100 м3/сут через {W}, м3/сут:')
    print(f'  истина {per_neighbor(n, "истина " + W)[0]:+.2f} | событийная оценка {obs[0]:+.2f} [{obs[1]:+.2f}; {obs[2]:+.2f}]'
          f' | CM {per_neighbor(n, "CM " + W)[0]:+.2f} | модель отклика {per_neighbor(n, "R " + W)[0]:+.2f}')
    print('Согласие с истиной по парам (Спирмен): '
          f'CM {spearmanr(n["CM " + W] * s, n["истина " + W] * s).correlation:+.2f}, '
          f'модель отклика {spearmanr(n["R " + W] * s, n["истина " + W] * s).correlation:+.2f}, '
          f'сами наблюдения {spearmanr(n[W] * s, n["истина " + W] * s).correlation:+.2f}')
    print(f'Модель отклика: tau = {rm.tau:.0f} сут, G_geo = {rm.g_geo:.3f}, G_f = {rm.g_f:.3f}')
    for r in info:
        print('  участок %s: связи CM против истинных — Спирмен %.2f; сумма f: CM %.2f, истина %.2f; '
              'медиана tau: CM %.0f, истина %.0f сут' % r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--blocks', nargs='+', default=['S1'])
    ap.add_argument('--step', type=int, default=7)
    a = ap.parse_args()
    blocks = load_blocks(a.data, a.blocks, a.step)
    run('сильный', (0.3, 0.9), blocks)
    run('слабый', (0.03, 0.10), blocks)


if __name__ == '__main__':
    main()
