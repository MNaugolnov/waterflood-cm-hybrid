"""Сквозной расчёт и сравнение вариантов на скользящих датах начала прогноза.

Для каждой даты начала прогноза (origin) всё обучается только на данных до неё, прогноз
строится на `horizon` шагов вперёд. Закачка и график работы скважин на прогнозном периоде
считаются заданными (это управляющие воздействия), фактические дебиты — неизвестными.

Варианты по жидкости:
    naive      — уровень последних 4 рабочих шагов x график работы (инерционный прогноз)
    cm_free    — CM, состояние на старте из свободного расчёта (как в исходной реализации)
    cm_anchor  — CM, состояние на старте привязано к факту
    cm_delta   — уровень + изменение по CM (CM «в приращениях»)
    hybrid     — уровень + нейросетевая поправка с признаками CM (гибрид)
    nn_only    — то же без признаков CM (показывает вклад физической модели)
Варианты по нефти:
    oil_naive    — уровень дебита нефти последних 4 рабочих шагов x график работы
    oil_const    — постоянная доля нефти (последние 8 шагов) x прогноз жидкости гибрида
    oil_dc       — ансамбль ХВ x прогноз жидкости гибрида (сквозной прогноз)
    oil_dc_fact  — ансамбль ХВ x фактическая жидкость (качество самих ХВ, не прогноз)
"""
from multiprocessing import Pool
import numpy as np
import pandas as pd
from .dc import DisplacementCurve
from .resnn import build_table, Stacker

CM_FEATURES = ('d_cm', 'd_cm_anchor', 'err0', 'fit_err', 'd_winj', 'tau', 'e', 'f_sum')
LIQ = ['naive', 'cm_free', 'cm_anchor', 'cm_delta', 'hybrid', 'nn_only']
OIL = ['oil_naive', 'oil_const', 'oil_dc', 'oil_dc_fact']


def wmape(fact, pred):
    s = np.abs(fact).sum()
    return np.abs(fact - pred).sum() / s if s > 0 else np.nan


def cum_err(fact, pred):
    s = fact.sum()
    return (pred.sum() - s) / s if s > 0 else np.nan


def _level(q_hist, up_hist, n=4):
    P = q_hist.shape[1]
    lvl = np.zeros(P)
    for j in range(P):
        act = np.where(up_hist[:, j] > 0.5)[0][-n:]
        lvl[j] = (q_hist[act, j] / up_hist[act, j]).mean() if len(act) else 0.0
    return lvl


def _oil_one(args):
    q_h, o_h, liq_fc, liq_fact, step = args
    act = np.where(q_h > 0)[0][-8:]
    fo = o_h[act].sum() / max(q_h[act].sum(), 1e-9) if len(act) else 0.0
    dc = DisplacementCurve().fit(q_h, o_h, step=step)
    return fo * liq_fc, dc.predict(liq_fc, step), dc.predict(liq_fact, step), dc.name


def prepare(blocks, origin_dates, horizon=26, verbose=True):
    """blocks: {имя: Dataset}. Для каждого участка — таблица признаков и базовые прогнозы."""
    prep = {}
    for name, ds in blocks.items():
        origins = [ds.index_of(d) for d in origin_dates]
        tab, bases, cms = build_table(ds, origins, horizon=horizon)
        prep[name] = {'ds': ds, 'tab': tab, 'bases': bases, 'cms': cms, 'origins': origins}
        if verbose:
            print(f'  участок {name}: добывающих {len(ds.prods)}, нагнетательных {len(ds.injs)}, '
                  f'строк обучающей таблицы {len(tab)}', flush=True)
    return prep


def evaluate(prep, origin_ids, horizon=26, min_hist=52, train_blocks=None, test_blocks=None,
             with_oil=True, n_jobs=2, verbose=True):
    """Сравнение вариантов. train_blocks/test_blocks позволяют проверить перенос сети на участок,
    которого не было в обучении (по умолчанию сеть учится на всех участках сразу)."""
    train_blocks = list(prep) if train_blocks is None else train_blocks
    test_blocks = list(prep) if test_blocks is None else test_blocks
    rows, forecasts = [], {}
    for oi in origin_ids:
        tr = pd.concat([prep[b]['tab'][(prep[b]['tab'].k < prep[b]['origins'][oi]) &
                                       (prep[b]['tab'].origin < prep[b]['origins'][oi])] for b in train_blocks])
        hybrid = Stacker(kind='mlp').fit(tr)
        nn_only = Stacker(kind='mlp', drop=CM_FEATURES).fit(tr)
        for b in test_blocks:
            c = prep[b]
            ds, t0, base = c['ds'], c['origins'][oi], c['bases'][c['origins'][oi]]
            te = c['tab'][c['tab'].origin == t0]
            n, P = te.k.nunique(), len(ds.prods)
            t1 = t0 + n
            up_h, q_h, o_h = ds.up[:t0], ds.liq[:t0], ds.oil[:t0]
            up_f, q_f, o_f = ds.up[t0:t1], ds.liq[t0:t1], ds.oil[t0:t1]
            pred = {k: base[k] for k in ('naive', 'cm_free', 'cm_anchor', 'cm_delta')}
            for nm, m in (('hybrid', hybrid), ('nn_only', nn_only)):
                r = m.predict(te).reshape(n, P)
                pred[nm] = np.maximum(base['level'] + r * base['qs'], 0.0) * up_f
            if with_oil:
                pred['oil_naive'] = _level(o_h, up_h) * up_f
                jobs = [(q_h[:, j], o_h[:, j], pred['hybrid'][:, j], q_f[:, j], ds.step) for j in range(P)]
                with Pool(n_jobs) as pool:
                    res = pool.map(_oil_one, jobs, chunksize=4)
                pred['oil_const'] = np.stack([r[0] for r in res], axis=1)
                pred['oil_dc'] = np.stack([r[1] for r in res], axis=1)
                pred['oil_dc_fact'] = np.stack([r[2] for r in res], axis=1)
                curves = [r[3] for r in res]
            for j, w in enumerate(ds.prods):
                hist_act = int((up_h[:, j] > 0.5).sum())
                fut_act = float((up_f[:, j] > 0.5).mean())
                row = {'block': b, 'origin': str(ds.dates[t0].date()), 'well': w,
                       'eligible': hist_act >= min_hist and fut_act >= 0.5,
                       'q_liq': q_f[:, j].mean(), 'q_oil': o_f[:, j].mean(),
                       'tau': c['cms'][t0].tau[j], 'cm_fit_err': c['cms'][t0].fit_err_[j]}
                for k in LIQ:
                    row[k] = wmape(q_f[:, j], pred[k][:, j])
                    row[k + '_cum'] = cum_err(q_f[:, j], pred[k][:, j])
                if with_oil:
                    row['curve'] = curves[j]
                    for k in OIL:
                        row[k] = wmape(o_f[:, j], pred[k][:, j])
                        row[k + '_cum'] = cum_err(o_f[:, j], pred[k][:, j])
                rows.append(row)
            forecasts[(b, oi)] = {'dates': ds.dates[t0:t1], 'liq': q_f, 'oil': o_f, 'pred': pred}
            if verbose:
                print(f'  дата {ds.dates[t0].date()}, участок {b}: готово', flush=True)
    return pd.DataFrame(rows), forecasts


def summarize(df, cols):
    e = df[df['eligible']]
    out = []
    for c in cols:
        v = e[c].dropna()
        out.append({'вариант': c, 'медиана wMAPE, %': 100 * v.median(), 'среднее wMAPE, %': 100 * v.mean(),
                    'доля скважин с wMAPE < 20 %': (v < 0.2).mean(),
                    'медиана |ошибки накопленной|, %': 100 * e[c + '_cum'].abs().median()})
    return pd.DataFrame(out).set_index('вариант')
