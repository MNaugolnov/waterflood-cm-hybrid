"""Расчётное ядро веб-приложения: обёртка над пакетом hybrid (CM + ХВ + нейросеть + модель отклика).

Интерфейс (app.py) вызывает только функции этого модуля; от Dash ядро не зависит,
поэтому его можно проверять и использовать отдельно.
"""
import dataclasses
import glob
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

import config
from . import reader

if not os.path.isdir(os.path.join(config.HYBRID_PATH, 'hybrid')):
    raise ImportError(
        'Не найден расчётный пакет hybrid. Ожидается папка "%s". Укажите путь в config.py '
        '(HYBRID_PATH) или в переменной окружения CRM_HYBRID_PATH.' % os.path.abspath(config.HYBRID_PATH))
sys.path.insert(0, os.path.abspath(config.HYBRID_PATH))

from hybrid.data import build_dataset                                   # noqa: E402
from hybrid.cm import CM                                                # noqa: E402
from hybrid.dc import DisplacementCurve                                 # noqa: E402
from hybrid.resnn import origin_block, build_table, Stacker             # noqa: E402
from hybrid.response import find_events, observed_response, model_response, potential   # noqa: E402
from hybrid.respmodel import ResponseModel, geo_weights                 # noqa: E402
from hybrid.evaluate import wmape, cum_err                              # noqa: E402


# ---------------------------------------------------------------- данные
def demo_blocks():
    """Примеры участков в папке данных: {подпись: (файл замеров, файл координат)}."""
    out = {}
    for d in sorted(glob.glob(os.path.join(config.DATA_PATH, 'Block *')),
                    key=lambda p: (len(os.path.basename(p)), os.path.basename(p))):
        b = os.path.basename(d).split(' ', 1)[1]
        fb = [f for f in glob.glob(os.path.join(d, '*')) if os.path.basename(f).lower() == f'block{b}.csv'.lower()]
        fc = sorted(glob.glob(os.path.join(d, 'coord*.csv')))
        if fb and fc:
            out[f'Участок {b}'] = (fb[0], fc[0])
    return out


def step_params(step):
    """Параметры, зависящие от шага расчёта (7 сут — суточные замеры, 30 сут — месячные данные)."""
    if step >= 20:
        return dict(gap=45, min_active=12, pre=3, sep=3, min_pts=1, anchor=3,
                    windows={'до': (-6, -3), '1-2 мес': (1, 3), '3-4 мес': (3, 5), '5-6 мес': (5, 7)})
    return dict(gap=None, min_active=26, pre=8, sep=8, min_pts=4, anchor=4,
                windows={'до': (-16, -8), '1-8 нед': (1, 9), '9-16 нед': (9, 17), '17-26 нед': (17, 26)})


@dataclasses.dataclass
class Model:
    name: str
    ds: object
    cm: object
    rm: object
    g: np.ndarray                 # [I,P] коэффициенты отклика, м3/сут жидкости на 1 м3/сут закачки
    info: dict
    calibrated: bool
    sp: dict
    coords: object = None         # координаты скважин (индекс — номер, столбцы X, Y)
    cache: dict = dataclasses.field(default_factory=dict)

    # --- состояние на конец истории ---
    @property
    def inj_now(self):
        return self.ds.inj[-4:].mean(axis=0)

    @property
    def up_now(self):
        return (self.ds.up[-4:].mean(axis=0) > 0.5).astype(float)

    @property
    def liq_now(self):
        return _level(self.ds.liq, self.ds.up)

    @property
    def oil_frac(self):
        """Доля нефти в жидкости по последним 8 рабочим шагам."""
        ds, out = self.ds, np.zeros(len(self.ds.prods))
        for j in range(len(ds.prods)):
            act = np.where(ds.liq[:, j] > 0)[0][-8:]
            out[j] = ds.oil[act, j].sum() / max(ds.liq[act, j].sum(), 1e-9) if len(act) else 0.0
        return out


def _level(q, up, n=4):
    out = np.zeros(q.shape[1])
    for j in range(q.shape[1]):
        act = np.where(up[:, j] > 0.5)[0][-n:]
        out[j] = (q[act, j] / up[act, j]).mean() if len(act) else 0.0
    return out


def default_response():
    d = config.RESPONSE_DEFAULT
    rm = ResponseModel(radius=d['radius'], f_cap=d['f_cap'])
    rm.tau, rm.g_geo, rm.g_f = d['tau'], d['g_geo'], d['g_f']
    rm.coef_ = np.array([rm.g_geo, rm.g_f])
    return rm


def calibrate_response(ds, log=None):
    """Калибровка модели отклика по загруженному участку. CM по сетке дат видит только прошлое."""
    cms, theta = {}, None
    for g in range(52, ds.T - 8, 13):
        cm = CM(ds.mask, ds.dist, step=ds.step).fit(ds.inj[:g], ds.up[:g], ds.liq[:g], theta0=theta,
                                                    maxiter=config.CM_MAXITER if theta is None else 300)
        theta, cms[g] = cm.theta, cm
    if not cms:
        return None, 'история короче 60 шагов'
    rm = ResponseModel(radius=config.RESPONSE_DEFAULT['radius'], f_cap=config.RESPONSE_DEFAULT['f_cap'])
    try:
        rm.fit([(ds, {g: c.f for g, c in cms.items()}, 52, ds.T)])
    except Exception as e:                                       # мало данных для регрессии
        return None, f'не удалось ({e})'
    if rm.n_ < 500 or (rm.g_geo + rm.g_f) <= 0:
        return None, f'мало наблюдений ({rm.n_}) или нулевой отклик'
    return rm, f'по {rm.n_} наблюдениям'


def build(df, coords, name='данные', step=None, calibrate=False, info=None):
    """Замеры + координаты -> настроенная модель участка."""
    info = dict(info or {})
    if step is None:
        step, info['median_gap'] = reader.detect_step(df[df['role'] == 'НЕФ'])
    sp = step_params(step)
    kw = dict(step=step, min_active_steps=sp['min_active'])
    if sp['gap']:
        kw.update(gap_prod=sp['gap'], gap_inj=sp['gap'])
    ds = build_dataset(df, coords, **kw)
    if not len(ds.prods) or not len(ds.injs):
        raise reader.DataError(
            f'После отбора осталось добывающих: {len(ds.prods)}, нагнетательных: {len(ds.injs)}. '
            f'Нужны координаты и не менее {sp["min_active"]} шагов работы у каждой скважины.')
    if ds.T < 40:
        raise reader.DataError(f'История слишком короткая: {ds.T} шагов по {step} сут (нужно не меньше 40).')
    cm = CM(ds.mask, ds.dist, step=ds.step).fit(ds.inj, ds.up, ds.liq, maxiter=config.CM_MAXITER)
    rm, calibrated, note = default_response(), False, ''
    if calibrate:
        if step != 7:
            note = 'калибровка доступна только для шага 7 сут — взяты коэффициенты по умолчанию'
        else:
            rm2, note = calibrate_response(ds)
            if rm2 is not None:
                rm, calibrated = rm2, True
            else:
                note = 'калибровка: ' + note + ' — взяты коэффициенты по умолчанию'
    q, _ = cm.simulate(ds.inj, ds.up)
    info.update(step=step, T=ds.T, n_prod=len(ds.prods), n_inj=len(ds.injs),
                date_min=ds.dates[0], date_max=ds.dates[-1] + pd.Timedelta(days=step - 1),
                cm_wmape=float(wmape(ds.liq, q)), cm_iter=int(cm.nit_), calib_note=note,
                has_oil=bool(ds.oil.sum() > 0))
    m = Model(name, ds, cm, rm, rm.gains(ds, cm.f), info, calibrated, sp, coords)
    return m


def load_demo(label, step=None, calibrate=False):
    fb, fc = demo_blocks()[label]
    return load_files(reader.read_path(fb), None, reader.read_path(fc), label, step, calibrate)


def load_files(prod_df, inj_df, coord_df, name, step=None, calibrate=False):
    df, coords, info = reader.assemble(prod_df, inj_df, coord_df, config.OIL_DENSITY)
    return build(df, coords, name, step, calibrate, info)


# ---------------------------------------------------------------- связи
def pairs_table(m, max_dist=None, min_coef=0.0, source='g', wells=None):
    """Пары «нагнетательная -> добывающая» с коэффициентами."""
    ds = m.ds
    ii, jj = np.where(m.g > 0) if source == 'g' else np.where(ds.mask)
    t = pd.DataFrame({'inj': [ds.injs[i] for i in ii], 'prod': [ds.prods[j] for j in jj],
                      'dist': ds.dist[ii, jj], 'g': m.g[ii, jj], 'f': m.cm.f[ii, jj], 'i': ii, 'j': jj})
    t['coef'] = t[source]
    if max_dist is not None:
        t = t[t['dist'] <= max_dist]
    t = t[t['coef'] >= min_coef]
    if wells:
        t = t[t['inj'].isin(wells) | t['prod'].isin(wells)]
    return t.sort_values('coef', ascending=False).reset_index(drop=True)


def injector_table(m):
    ds = m.ds
    work = m.up_now
    near = (ds.dist <= m.rm.radius) & (ds.dist > 1.0)
    return pd.DataFrame({
        'inj': ds.injs, 'rate': m.inj_now, 'n_near': near.sum(axis=1),
        'sum_g': (m.g * work).sum(axis=1), 'sum_f': m.cm.f.sum(axis=1),
        'eff_oil': (m.g * work * m.oil_frac).sum(axis=1),
    })


# ---------------------------------------------------------------- прогноз
def _extend(ds, n, inj_future=None):
    """Участок, продлённый на n шагов: закачка и график работы — как в конце истории."""
    if n <= 0:
        return ds
    T = ds.T
    inj_f = np.repeat(ds.inj[-4:].mean(axis=0)[None], n, axis=0) if inj_future is None else inj_future
    up_f = np.repeat((ds.up[-4:].mean(axis=0) > 0.5).astype(float)[None], n, axis=0)
    dates = ds.dates.append(pd.date_range(ds.dates[-1] + pd.Timedelta(days=ds.step), periods=n, freq=f'{ds.step}D'))
    z = np.zeros((n, len(ds.prods)))
    return dataclasses.replace(
        ds, dates=dates, liq=np.vstack([ds.liq, z]), oil=np.vstack([ds.oil, z]), up=np.vstack([ds.up, up_f]),
        inj=np.vstack([ds.inj, inj_f]), bhp=np.vstack([ds.bhp, np.full_like(z, np.nan)])), T


def _truncate(ds, t0):
    return dataclasses.replace(ds, dates=ds.dates[:t0], liq=ds.liq[:t0], oil=ds.oil[:t0], up=ds.up[:t0],
                               inj=ds.inj[:t0], bhp=ds.bhp[:t0])


def _cm_at(m, t0):
    if t0 >= m.ds.T:
        return m.cm
    key = ('cm', t0)
    if key not in m.cache:
        ds = m.ds
        m.cache[key] = CM(ds.mask, ds.dist, step=ds.step).fit(
            ds.inj[:t0], ds.up[:t0], ds.liq[:t0], theta0=None, maxiter=config.CM_MAXITER)
    return m.cache[key]


def forecast(m, t0, horizon=26, use_nn=False, with_oil=True):
    """Прогноз от шага t0 на horizon шагов.

    t0 < T — проверка на истории: модели видят только данные до t0, закачка и график работы
    на прогнозном периоде берутся фактические. Часть периода за концом истории (и весь период
    при t0 = T) считается при закачке и графике работы, замороженных на последнем уровне.
    """
    key = ('fc', t0, horizon, use_nn, with_oil)
    if key in m.cache:
        return m.cache[key]
    ds = m.ds
    t0 = int(min(max(t0, 30), ds.T))
    t1 = t0 + horizon
    ext = _extend(ds, t1 - ds.T)
    dse = ext[0] if isinstance(ext, tuple) else ext
    cm = _cm_at(m, t0)
    te, base = origin_block(dse, cm, t0, t1, horizon=horizon)
    n, P = t1 - t0, len(ds.prods)
    up_f = dse.up[t0:t1]
    pred = {'naive': base['naive'], 'cm': base['cm_delta']}
    notes = []
    if use_nn:
        first = int(max(30, min(78, t0 // 2)))
        if t0 - first < 3 * horizon:
            notes.append('Нейросетевая поправка не построена: мало истории до даты прогноза.')
        else:
            tab, _, _ = build_table(_truncate(ds, t0), [], horizon=horizon, first=first, every=8)
            st = Stacker(kind='mlp').fit(tab)
            r = st.predict(te).reshape(n, P)
            pred['hybrid'] = np.maximum(base['level'] + r * base['qs'], 0.0) * up_f
            notes.append(f'Нейросетевая поправка обучена на {st.n_samples_} примерах (только данные до даты прогноза).')
    best = 'hybrid' if 'hybrid' in pred else 'cm'
    # базовый фонд: скважины, работавшие на дату прогноза (не меньше 2 из 4 последних шагов).
    # Скважины, запущенные позже, в суммы и оценки не входят: их уровень на дату прогноза неизвестен.
    fund = (ds.up[t0 - 4:t0] > 0.5).sum(axis=0) >= 2
    oil, curves = {}, None
    if with_oil and m.info.get('has_oil'):
        o_h, q_h = ds.oil[:t0], ds.liq[:t0]
        oil['naive'] = _level(o_h, ds.up[:t0]) * up_f
        dc_pred, curves = np.zeros((n, P)), [''] * P
        idx = np.where(fund)[0]
        out = _dc_all([(q_h[:, j], o_h[:, j], pred[best][:, j], ds.step) for j in idx])
        for j, (p, name) in zip(idx, out):
            dc_pred[:, j], curves[j] = p, name
        oil['dc'] = dc_pred
    n_fact = max(min(ds.T, t1) - t0, 0)
    res = {'t0': t0, 't1': t1, 'n_fact': n_fact, 'dates': dse.dates[t0:t1], 'pred': pred, 'oil': oil,
           'best': best, 'curves': curves, 'notes': notes, 'up_f': up_f, 'horizon': horizon, 'fund': fund,
           'metrics': _metrics(ds, t0, n_fact, pred, oil, fund)}
    m.cache[key] = res
    return res


def _dc_one(args):
    q_h, o_h, liq_fc, step = args
    dc = DisplacementCurve().fit(q_h, o_h, step=step)
    return dc.predict(liq_fc, step), dc.name


def _dc_all(jobs):
    """ХВ по скважинам. Подбор кривых — около секунды на скважину, поэтому на больших участках
    считается в нескольких процессах; при любой ошибке запуска процессов — последовательно."""
    n = config.JOBS or min(4, os.cpu_count() or 1)
    if n > 1 and len(jobs) >= 8:
        try:
            with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context('spawn')) as ex:
                return list(ex.map(_dc_one, jobs, chunksize=2))
        except Exception:
            pass
    return [_dc_one(j) for j in jobs]


def _metrics(ds, t0, n_fact, pred, oil, fund):
    """Ошибки на той части прогноза, где есть факт: по базовому фонду в сумме и медиана по скважинам."""
    if n_fact < 4:
        return None
    sl = slice(t0, t0 + n_fact)
    ok = fund & ((ds.up[:t0] > 0.5).sum(axis=0) >= 26)
    rows = []
    for kind, fact, preds in (('жидкость', ds.liq[sl], pred), ('нефть', ds.oil[sl], oil)):
        for k, p in preds.items():
            p = p[:n_fact]
            w = [wmape(fact[:, j], p[:, j]) for j in np.where(ok)[0] if fact[:, j].sum() > 0]
            rows.append({'kind': kind, 'method': k,
                         'block_wmape': wmape(fact[:, fund].sum(axis=1), p[:, fund].sum(axis=1)),
                         'block_cum': cum_err(fact[:, fund].sum(axis=1), p[:, fund].sum(axis=1)),
                         'well_wmape': float(np.median(w)) if w else np.nan, 'n_wells': len(w)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- отклик на закачку
def _per_neighbor(df, col, n_boot=2000):
    """Средний отклик одной соседней скважины на +100 м3/сут закачки; бутстреп по событиям."""
    v = df.dropna(subset=[col])
    g = v.groupby('ev').agg(s=(col, 'sum'), k=(col, 'size'), d=('d', 'first'))
    num, den = (g.s * np.sign(g.d)).values, (g.k * g.d.abs()).values
    if len(num) < 5:
        return np.nan, np.nan, np.nan, len(g)
    idx = np.random.default_rng(0).integers(0, len(num), (n_boot, len(num)))
    bs = 100 * num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return 100 * num.sum() / den.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(g)


def event_study(m):
    """Фактический отклик добычи на резкие изменения закачки и то же по моделям."""
    if 'events' in m.cache:
        return m.cache['events']
    ds, sp = m.ds, m.sp
    W = sp['windows']
    fwd = [w for w, (a, _) in W.items() if a > 0]
    ev = find_events(ds, pre=sp['pre'], post=sp['pre'], sep=sp['sep'])
    res = {'events': ev, 'summary': None, 'windows': fwd, 'n_events': len(ev)}
    if len(ev):
        kw = dict(pre=sp['pre'], windows=W, min_pts=sp['min_pts'])
        t = observed_response(ds, ev, **kw)
        if len(t):
            hz = max(z for _, z in W.values())
            oil = observed_response(ds, ev, series=ds.oil, **kw)[['ev', 'j'] + list(W)]
            t = t.merge(oil.rename(columns={w: 'нефть ' + w for w in W}), on=['ev', 'j'])
            cm_r = model_response(lambda inj, up: m.cm.simulate(inj, up)[1], ds, ev, windows=W, horizon=hz)
            rm_r = model_response(m.rm.simulator(ds, m.cm.f), ds, ev, windows=W, horizon=hz)
            t = t.merge(cm_r.rename(columns={'мод ' + w: 'CM ' + w for w in fwd}), on=['ev', 'j'], how='left')
            t = t.merge(rm_r.rename(columns={'мод ' + w: 'R ' + w for w in fwd}), on=['ev', 'j'], how='left')
            near = t[t.dist <= m.rm.radius]
            rows = []
            for w in fwd:
                v = near.dropna(subset=[w])
                f = _per_neighbor(v, w)
                o = _per_neighbor(near, 'нефть ' + w)
                rows.append({'window': w, 'fact': f[0], 'lo': f[1], 'hi': f[2], 'n_ev': f[3], 'n_pairs': len(v),
                             'oil': o[0], 'oil_lo': o[1], 'oil_hi': o[2],
                             'cm': _per_neighbor(v, 'CM ' + w)[0], 'rm': _per_neighbor(v, 'R ' + w)[0]})
            res['summary'] = pd.DataFrame(rows)
        ev = ev.copy()
        ev['injector'] = [ds.injs[i] for i in ev.i]
        ev['date'] = [ds.dates[k].date() for k in ev.t]
        res['events'] = ev
    m.cache['events'] = res
    return res


# ---------------------------------------------------------------- сценарий перераспределения
def manual_delta(m, donors, recipients, volume):
    """Снять volume м3/сут с доноров (пропорционально текущей закачке) и отдать получателям (поровну)."""
    ds, d = m.ds, np.zeros(len(m.ds.injs))
    now = m.inj_now
    di = [ds.injs.index(w) for w in (donors or []) if w in ds.injs]
    ri = [ds.injs.index(w) for w in (recipients or []) if w in ds.injs and w not in (donors or [])]
    taken = 0.0
    if di:
        cap = now[di].sum()
        taken = min(float(volume), cap)
        if cap > 0:
            d[di] -= taken * now[di] / cap
    give = taken if di else float(volume)
    if ri:
        d[ri] += give / len(ri)
    return d


def suggest_delta(m, limit=0.2, target='oil'):
    """Перераспределение при неизменной суммарной закачке: каждая нагнетательная меняется не больше
    чем на limit от текущей приёмистости; максимум ожидаемого прироста нефти (или жидкости).
    Задача линейная, решение — снять закачку с наименее эффективных и отдать наиболее эффективным."""
    now = m.inj_now
    w = m.up_now * (m.oil_frac if target == 'oil' else 1.0)
    eff = (m.g * w).sum(axis=1)
    room = 2.0 * limit * now                       # сколько можно добавить, стартуя с нижней границы
    d = -limit * now
    budget = limit * now.sum()
    for i in np.argsort(-eff):
        add = min(room[i], budget)
        d[i] += add
        budget -= add
        if budget <= 1e-9:
            break
    return d


def scenario(m, delta, horizon=26):
    """Эффект изменения закачки на delta [I] м3/сут (относительно текущего уровня) на horizon шагов."""
    ds = m.ds
    now, up_now = m.inj_now, m.up_now
    base_f = np.repeat(now[None], horizon, axis=0)
    new_f = np.maximum(base_f + np.asarray(delta, float)[None], 0.0)
    real = new_f[0] - now                                             # с учётом ограничения «не меньше нуля»
    base, new = np.vstack([ds.inj, base_f]), np.vstack([ds.inj, new_f])
    d_liq = m.rm.effect(ds, base, new, m.cm.f)[ds.T:] * up_now        # [h,P]
    d_oil = d_liq * m.oil_frac
    up = np.vstack([ds.up, np.repeat(up_now[None], horizon, axis=0)])
    d_cm = (m.cm.simulate(new, up)[1] - m.cm.simulate(base, up)[1])[ds.T:] * up_now
    dates = pd.date_range(ds.dates[-1] + pd.Timedelta(days=ds.step), periods=horizon, freq=f'{ds.step}D')
    return {'dates': dates, 'delta': real, 'd_liq': d_liq, 'd_oil': d_oil, 'd_cm': d_cm,
            'liq_end': float(d_liq[-1].sum()), 'oil_end': float(d_oil[-1].sum()),
            'oil_cum': float(d_oil.sum() * ds.step), 'cm_end': float(d_cm[-1].sum()),
            'inj_change': float(real.sum()), 'moved': float(np.abs(real).sum() / 2.0), 'horizon': horizon}
