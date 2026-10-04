"""Отклик добычи на изменения закачки: измерение по фактическим данным и проверка моделей.

1. find_events        — резкие устойчивые изменения приёмистости (остановы, пуски, ступени).
2. observed_response  — фактическое изменение дебита соседних добывающих после события
                        («разность разностей»: за вычетом изменения у дальних скважин блока).
3. model_response     — то же по модели: расчёт с фактической закачкой минус расчёт, в котором
                        закачка этой нагнетательной оставлена на уровне до события.
4. response_ratio     — доля изменения закачки, проявившаяся в добыче соседей, с бутстреп-интервалом.

Потенциальный дебит — дебит в пересчёте на полное время работы (q / u); шаги с простоем
больше половины шага в расчёт не входят.
"""
import numpy as np
import pandas as pd

WINDOWS = {'до': (-16, -8), '1-8 нед': (1, 9), '9-16 нед': (9, 17), '17-26 нед': (17, 26)}


def potential(q, up):
    return np.where(up > 0.5, q / np.maximum(up, 1e-6), np.nan)


def find_events(ds, pre=8, post=8, min_abs=50.0, min_rel=0.4, sep=8):
    """События по нагнетательным: |среднее за post шагов после - среднее за pre шагов до|
    не меньше min_abs м3/сут и не меньше min_rel от большего из двух уровней."""
    T, n_inj = ds.inj.shape
    out = []
    for i in range(n_inj):
        x = ds.inj[:, i]
        cs = np.concatenate([[0.0], np.cumsum(x)])
        t_all = np.arange(pre, T - post)
        lvl_pre = (cs[t_all] - cs[t_all - pre]) / pre
        lvl_post = (cs[t_all + post] - cs[t_all]) / post
        d = lvl_post - lvl_pre
        ok = (np.abs(d) >= min_abs) & (np.abs(d) / np.maximum(np.maximum(lvl_pre, lvl_post), 1e-9) >= min_rel)
        k = 0
        while k < len(t_all):
            if ok[k]:
                seg = np.arange(k, min(k + sep, len(t_all)))
                seg = seg[ok[seg]]
                kb = seg[np.argmax(np.abs(d[seg]))]
                a, b = lvl_pre[kb], lvl_post[kb]
                kind = 'останов' if b < 0.1 * a else 'пуск' if a < 0.1 * b else 'рост' if b > a else 'снижение'
                out.append({'i': i, 't': int(t_all[kb]), 'd': d[kb], 'pre': a, 'post': b, 'type': kind})
                k = kb + sep
            else:
                k += 1
    ev = pd.DataFrame(out)
    ev['ev'] = np.arange(len(ev))
    return ev


def observed_response(ds, ev, series=None, up=None, pre=8, windows=WINDOWS, ctrl_radius=1500.0, min_pts=4,
                      stable_only=True, lead=4, min_up=0.9):
    """Фактический отклик по парам «событие — добывающая», м3/сут, за вычетом контроля.

    series — ряд дебитов [T,P] (по умолчанию жидкость). Контроль — медианное относительное
    изменение у скважин дальше ctrl_radius от нагнетательной в том же окне.

    stable_only: в расчёт берутся только скважины, работавшие без остановок (доля работы не ниже
    min_up) от lead шагов до базового периода и до конца окна. Без этого оценка смещена: остановки
    добывающих часто совпадают по времени с изменениями закачки, а после пуска дебит временно
    завышен (проверено на полусинтетике, смещение около -0.07 от изменения закачки)."""
    q = ds.liq if series is None else series
    up_ = ds.up if up is None else up
    pot = potential(q, up_)
    bad = np.concatenate([np.zeros((1, up_.shape[1])), np.cumsum(up_ < min_up, axis=0)])   # число «плохих» шагов до k
    rows = []
    for e in ev.itertuples():
        t, dist = e.t, ds.dist[e.i]
        base = pot[t - pre:t]
        n_ok = np.isfinite(base).sum(axis=0)
        q0 = np.nanmean(np.where(n_ok >= pre // 2 + 1, base, np.nan), axis=0)
        rec = {'ev': e.ev, 'i': e.i, 't': t, 'd': e.d, 'type': e.type}
        for name, (a, z) in windows.items():
            w = pot[max(t + a, 0):min(t + z, ds.T)]
            m = np.where(np.isfinite(w).sum(axis=0) >= min_pts, np.nanmean(w, axis=0), np.nan) if len(w) else np.full(len(q0), np.nan)
            if stable_only:
                lo, hi = max(min(t + a, t - pre) - lead, 0), min(max(t + z, t), ds.T)
                m = np.where(bad[hi] - bad[lo] == 0, m, np.nan)
            rel = m / q0 - 1.0
            far = dist > ctrl_radius
            ctrl = np.nanmedian(rel[far]) if np.isfinite(rel[far]).sum() >= 3 else 0.0
            rec[name] = (rel - ctrl) * q0
        for j in np.where(np.isfinite(q0))[0]:
            row = {k: v for k, v in rec.items() if k not in windows}
            row.update({'j': j, 'dist': dist[j], 'q0': q0[j]})
            row.update({name: rec[name][j] for name in windows})
            rows.append(row)
    return pd.DataFrame(rows)


def model_response(simulate, ds, ev, windows=WINDOWS, horizon=26):
    """Отклик по модели для тех же событий.

    simulate(inj, up) -> потенциальный дебит s[T,P]. Для каждого события считается разность
    расчётов с фактической закачкой и с закачкой события, оставленной на уровне до него."""
    rows = []
    for e in ev.itertuples():
        t1 = min(e.t + horizon, ds.T)
        s_a = simulate(ds.inj[:t1], ds.up[:t1])
        inj_cf = ds.inj[:t1].copy()
        inj_cf[e.t:, e.i] = e.pre
        dp = s_a - simulate(inj_cf, ds.up[:t1])
        for name, (a, z) in windows.items():
            if a < 0:
                continue
            w = dp[e.t + a:min(e.t + z, t1)]
            for j in range(dp.shape[1]):
                rows.append({'ev': e.ev, 'j': j, 'win': name, 'pred': w[:, j].mean() if len(w) else np.nan})
    p = pd.DataFrame(rows).pivot_table(index=['ev', 'j'], columns='win', values='pred').reset_index()
    return p.rename(columns={w: 'мод ' + w for w in windows})


def response_ratio(tab, col, radius=1000.0, n_boot=2000, seed=0):
    """Сумма отклика соседей (в радиусе) / сумма |изменения закачки|, по событиям; бутстреп по событиям."""
    near = tab[tab['dist'] <= radius]
    g = near.groupby('ev').agg(v=(col, lambda s: s.sum(min_count=1)), d=('d', 'first')).dropna()
    num, den = (g['v'] * np.sign(g['d'])).values, g['d'].abs().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    bs = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return num.sum() / den.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(num)
