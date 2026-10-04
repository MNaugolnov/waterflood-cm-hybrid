"""Отклик добычи на изменения закачки: измерение, проверка CM и калиброванная модель отклика.

Пример:
    python run_response.py --data data --blocks S1 --out results

Что делает:
1. находит резкие изменения приёмистости (остановы, пуски, ступени);
2. измеряет фактический отклик соседних добывающих (жидкость, нефть) с интервалами;
3. сравнивает с откликом по CM, обученной только на данных до события;
4. обучает калиброванную модель отклика на последовательных периодах и проверяет её на
   следующих за ними событиях;
5. сохраняет итоговые коэффициенты отклика g (м3/сут жидкости на 1 м3/сут закачки) по участкам.
"""
import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from hybrid.cm import CM
from hybrid.response import find_events, observed_response, model_response, WINDOWS
from hybrid.respmodel import ResponseModel
from run_hybrid import load_blocks

warnings.filterwarnings('ignore')
FWD = [w for w, (a, _) in WINDOWS.items() if a > 0]


def per_neighbor(df, col, n_boot=2000):
    """Средний отклик одной соседней скважины на +100 м3/сут закачки, м3/сут; бутстреп по событиям."""
    v = df.dropna(subset=[col])
    g = v.groupby('ev').agg(s=(col, 'sum'), k=(col, 'size'), d=('d', 'first'))
    num, den = (g.s * np.sign(g.d)).values, (g.k * g.d.abs()).values
    if len(num) < 5:
        return np.nan, np.nan, np.nan, len(g), len(v)
    idx = np.random.default_rng(0).integers(0, len(num), (n_boot, len(num)))
    bs = 100 * num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return 100 * num.sum() / den.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(g), len(v)


def skill(df, fact, pred, trim=None):
    """1 - MSE/MSE(нулевой отклик) по событиям (сумма по соседям). trim — отсечь долю самых больших |факт|."""
    g = df.dropna(subset=[fact, pred]).groupby('ev')[[fact, pred]].sum()
    if trim:
        g = g[g[fact].abs() <= g[fact].abs().quantile(1 - trim)]
    return 1.0 - ((g[fact] - g[pred]) ** 2).mean() / (g[fact] ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--blocks', nargs='+', default=['S1'])
    ap.add_argument('--step', type=int, default=7)
    ap.add_argument('--radius', type=float, default=1000.0, help='радиус соседства, м')
    ap.add_argument('--folds', nargs='+', default=['2017-01-01', '2017-07-01', '2018-01-01', '2018-07-01', '2019-01-01'],
                    help='даты раздела: модель учится до даты, проверяется на событиях до следующей даты')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    blocks = load_blocks(a.data, a.blocks, a.step)

    print('CM по сетке дат (каждая модель видит только прошлое)...')
    F, events, tabs = {}, {}, []
    for b, ds in blocks.items():
        cms, theta = {}, None
        for g in range(52, ds.T - 8, 13):
            cm = CM(ds.mask, ds.dist, step=ds.step).fit(ds.inj[:g], ds.up[:g], ds.liq[:g], theta0=theta,
                                                        maxiter=2000 if theta is None else 400)
            theta, cms[g] = cm.theta, cm
        F[b] = {g: cm.f for g, cm in cms.items()}
        ev = find_events(ds)
        events[b] = ev
        t = observed_response(ds, ev)
        oil = observed_response(ds, ev, series=ds.oil)[['ev', 'j'] + list(WINDOWS)]
        t = t.merge(oil.rename(columns={w: 'нефть ' + w for w in WINDOWS}), on=['ev', 'j'])
        parts = [model_response(lambda inj, up, cm=cm: cm.simulate(inj, up)[1], ds, ev[(ev.t >= g) & (ev.t < g + 13)])
                 for g, cm in cms.items() if ((ev.t >= g) & (ev.t < g + 13)).any()]
        t = t.merge(pd.concat(parts).rename(columns={'мод ' + w: 'CM ' + w for w in FWD}), on=['ev', 'j'], how='left')
        t['well'] = [ds.prods[j] for j in t.j]
        t['injector'] = [ds.injs[i] for i in t.i]
        t['date'] = [ds.dates[k].date() for k in t.t]
        t['block'], t['ev'] = b, b + '-' + t.ev.astype(str)
        tabs.append(t)
        print(f'  участок {b}: событий {len(ev)}', flush=True)
    tab = pd.concat(tabs, ignore_index=True)
    near = tab[tab.dist <= a.radius]

    # ---------- 1. фактический отклик ----------
    rows = []
    for w in WINDOWS:
        for name, col in (('жидкость', w), ('нефть', 'нефть ' + w)):
            m, lo, hi, n_ev, n_pair = per_neighbor(near, col)
            rows.append({'выборка': 'все события', 'окно': w, 'показатель': name, 'отклик': m, 'нижн. 95 %': lo,
                         'верхн. 95 %': hi, 'событий': n_ev, 'пар': n_pair})
    for ty, g in near.groupby('type'):
        for name, col in (('жидкость', '17-26 нед'), ('нефть', 'нефть 17-26 нед')):
            m, lo, hi, n_ev, n_pair = per_neighbor(g, col)
            rows.append({'выборка': ty, 'окно': '17-26 нед', 'показатель': name, 'отклик': m, 'нижн. 95 %': lo,
                         'верхн. 95 %': hi, 'событий': n_ev, 'пар': n_pair})
    fact = pd.DataFrame(rows)
    fact.round(3).to_csv(os.path.join(a.out, 'response_observed.csv'), index=False, encoding='utf-8-sig')

    # ---------- 2. CM против факта ----------
    rows = []
    for w in FWD:
        v = near.dropna(subset=[w, 'CM ' + w])
        s = np.sign(v.d)
        rows.append({'окно': w, 'факт': per_neighbor(v, w)[0], 'CM': per_neighbor(v, 'CM ' + w)[0],
                     'Спирмен по парам': spearmanr(v['CM ' + w] * s, v[w] * s).correlation,
                     'качество по событиям': skill(v, w, 'CM ' + w), 'пар': len(v)})
    cm_tab = pd.DataFrame(rows)

    # ---------- 3. модель отклика на последовательных периодах ----------
    print('Модель отклика: обучение до даты, проверка на следующих событиях...')
    test, params = [], []
    for k, sp in enumerate(a.folds):
        items = [(ds, F[b], 52, ds.index_of(sp)) for b, ds in blocks.items()]
        rm = ResponseModel(radius=a.radius).fit(items)
        params.append({'обучение до': sp, 'tau, сут': rm.tau, 'G_geo': rm.g_geo, 'G_f': rm.g_f})
        for b, ds in blocks.items():
            lo = ds.index_of(sp)
            hi = ds.index_of(a.folds[k + 1]) if k + 1 < len(a.folds) else ds.T
            ev = events[b][(events[b].t >= lo) & (events[b].t < hi)]
            if not len(ev):
                continue
            p = pd.concat([model_response(rm.simulator(ds, F[b], e.t), ds, ev[ev.ev == e.ev]) for e in ev.itertuples()])
            p = p.rename(columns={'мод ' + w: 'R ' + w for w in FWD})
            p['ev'] = b + '-' + p.ev.astype(str)
            p['fold'] = sp
            test.append(p)
    test = near.merge(pd.concat(test), on=['ev', 'j'])
    rows = []
    for w in FWD:
        v = test.dropna(subset=[w, 'CM ' + w, 'R ' + w])
        s = np.sign(v.d)
        m, lo, hi, n_ev, n_pair = per_neighbor(v, w)
        rows.append({'окно': w, 'событий': n_ev, 'факт': m, 'нижн. 95 %': lo, 'верхн. 95 %': hi,
                     'CM': per_neighbor(v, 'CM ' + w)[0], 'модель отклика': per_neighbor(v, 'R ' + w)[0],
                     'качество CM': skill(v, w, 'CM ' + w), 'качество модели отклика': skill(v, w, 'R ' + w),
                     'качество CM (без 5 % выбросов)': skill(v, w, 'CM ' + w, 0.05),
                     'качество модели отклика (без 5 % выбросов)': skill(v, w, 'R ' + w, 0.05),
                     'Спирмен CM': spearmanr(v['CM ' + w] * s, v[w] * s).correlation,
                     'Спирмен модели отклика': spearmanr(v['R ' + w] * s, v[w] * s).correlation})
    held = pd.DataFrame(rows)
    held.round(3).to_csv(os.path.join(a.out, 'response_heldout.csv'), index=False, encoding='utf-8-sig')

    # ---------- 4. итоговая модель на всех данных ----------
    rm = ResponseModel(radius=a.radius).fit([(ds, F[b], 52, ds.T) for b, ds in blocks.items()])
    inj_rows = []
    for b, ds in blocks.items():
        g = rm.gains(ds, F[b])
        pd.DataFrame(g, index=ds.injs, columns=ds.prods).round(4).to_csv(
            os.path.join(a.out, f'response_gains_block{b}.csv'), encoding='utf-8-sig')
        # сводка по нагнетательным: сколько жидкости и нефти даёт 1 м3/сут закачки (установившийся отклик)
        last = slice(ds.T - 8, ds.T)
        oil_cut = ds.oil[last].sum(axis=0) / np.maximum(ds.liq[last].sum(axis=0), 1e-9)
        f_last = F[b][max(F[b])]
        near_ok = (ds.dist <= a.radius) & (ds.dist > 1.0)
        for i, w in enumerate(ds.injs):
            inj_rows.append({'участок': b, 'нагнетательная': w, 'закачка, м3/сут (посл. 8 нед)': ds.inj[last, i].mean(),
                             'соседей в радиусе': int(near_ok[i].sum()),
                             'жидкости на 1 м3 закачки (модель отклика)': g[i].sum(),
                             'нефти на 1 м3 закачки (модель отклика)': (g[i] * oil_cut).sum(),
                             'жидкости на 1 м3 закачки (CM, сумма f)': (f_last[i] * near_ok[i]).sum(),
                             'нефти на 1 м3 закачки (CM)': (f_last[i] * near_ok[i] * oil_cut).sum()})
    inj_tab = pd.DataFrame(inj_rows)
    inj_tab.round(4).to_csv(os.path.join(a.out, 'response_injectors.csv'), index=False, encoding='utf-8-sig')
    keep = ['block', 'ev', 'date', 'injector', 'type', 'd', 'well', 'dist', 'q0'] + list(WINDOWS) + \
           ['нефть ' + w for w in WINDOWS] + ['CM ' + w for w in FWD]
    near[keep].round(2).to_csv(os.path.join(a.out, 'response_events.csv'), index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 220)
    print(f'\nСобытий: {tab.ev.nunique()}. Отклик одной соседней скважины (до {a.radius:.0f} м, без остановок) '
          f'на +100 м3/сут закачки, м3/сут:')
    print(fact.round(2).to_string(index=False))
    print(f'\nСредний дебит жидкости такой скважины: {near.q0.mean():.0f} м3/сут')
    print('\nCM против факта (все события, CM обучена до события):')
    print(cm_tab.round(3).to_string(index=False))
    print('\nПараметры модели отклика по периодам обучения:')
    print(pd.DataFrame(params).round(3).to_string(index=False))
    print('\nПроверка на событиях после даты обучения:')
    print(held.round(3).T.to_string(header=False))
    print(f'\nИтоговая модель: tau = {rm.tau:.0f} сут, G_geo = {rm.g_geo:.3f}, G_f = {rm.g_f:.3f}')
    act = inj_tab[inj_tab.iloc[:, 2] > 10]
    print('Работающие нагнетательные: коэффициенты «на 1 м3/сут закачки» (медиана; 10-90 %):')
    for c in inj_tab.columns[4:]:
        print(f'  {c}: {act[c].median():.3f} ({act[c].quantile(0.1):.3f}-{act[c].quantile(0.9):.3f})')
    print(f'Время расчёта: {time.time() - t_start:.0f} с. Результаты: {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
