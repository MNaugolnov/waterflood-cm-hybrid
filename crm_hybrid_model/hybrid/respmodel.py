"""Модель отклика на перераспределение закачки, калиброванная по фактическим изменениям.

Зачем отдельная модель. CM настраивается на уровень добычи и поэтому приписывает закачке почти
всю добычу (сумма f по нагнетательной 0.8-1.0, tau упирается в верхнюю границу). Фактический
отклик на изменение закачки на проверенных участках в 2-3 раза слабее, а самые большие f
оказываются ложными. Для задачи перераспределения нужен прирост, а не уровень, поэтому здесь
коэффициенты оцениваются только по изменениям:

    d q_j(окно) = sum_i g_ij * d c_i(окно),      c_i = закачка, сглаженная звеном 1-го порядка (tau)
    g_ij = G_geo * w_ij + G_f * min(f_ij, f_cap)

    w_ij  — геометрические доли (обратно пропорционально расстоянию, сумма по добывающим = 1);
    f_ij  — связи из CM; используются как подсказка, какие пары связаны, но их величина
            ограничивается (f_cap) и перекалибруется общим множителем G_f;
    G_geo, G_f, tau — общие для всех участков, оцениваются по всем изменениям закачки сразу.

Четыре общих параметра вместо тысяч парных: по отдельной паре сигнал в данных слишком слаб
(проверено на полусинтетике), а общий уровень отклика определяется надёжно.

Оценка — регрессия с фиксированными эффектами «участок x время» (из дебита и из закачки
вычитается среднее по участку в то же окно), только по скважинам, работавшим без остановок.
"""
import numpy as np
from scipy.optimize import nnls
from .response import potential, WINDOWS

FWD = {k: v for k, v in WINDOWS.items() if v[0] > 0}


def lag_filter(inj, tau, step=7.0):
    """Звено первого порядка: c_k = a c_{k-1} + (1 - a) I_k."""
    a = np.exp(-step / tau)
    c = np.empty_like(inj, dtype=float)
    c[0] = inj[0]
    for k in range(1, len(inj)):
        c[k] = a * c[k - 1] + (1.0 - a) * inj[k]
    return c


def geo_weights(ds, radius=1000.0):
    w = np.where((ds.dist <= radius) & (ds.dist > 1.0), 1.0 / np.maximum(ds.dist, 100.0), 0.0)
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)


def _changes(x, t_list, pre, windows, min_pts=4, bad=None, lead=4):
    """Изменение ряда: среднее в окне минус среднее за pre шагов до t. Форма [окно, время, столбец]."""
    T = x.shape[0]
    out = np.full((len(windows), len(t_list), x.shape[1]), np.nan)
    for n, t in enumerate(t_list):
        base = x[t - pre:t]
        ok0 = np.isfinite(base).sum(axis=0) >= pre // 2 + 1
        b = np.where(ok0, np.nanmean(np.where(np.isfinite(base), base, np.nan), axis=0), np.nan)
        for k, (a, z) in enumerate(windows.values()):
            w = x[t + a:min(t + z, T)]
            if len(w) < min_pts:
                continue
            m = np.where(np.isfinite(w).sum(axis=0) >= min_pts, np.nanmean(w, axis=0), np.nan)
            if bad is not None:
                lo, hi = max(t - pre - lead, 0), min(t + z, T)
                m = np.where(bad[hi] - bad[lo] == 0, m, np.nan)
            out[k, n] = m - b
    return out


class ResponseModel:
    def __init__(self, radius=1000.0, taus=(15.0, 30.0, 60.0, 120.0, 240.0, 480.0), f_cap=0.15,
                 pre=8, every=4, min_up=0.9, use_f=True):
        self.radius, self.taus, self.f_cap = radius, taus, f_cap
        self.pre, self.every, self.min_up, self.use_f = pre, every, min_up, use_f

    # ---------- строки регрессии для одного участка ----------
    @staticmethod
    def _f_at(f, t):
        """f — матрица либо словарь {шаг: матрица} со связями CM, обученной на данных до этого шага.
        Для момента t берётся последняя модель, обученная не позже t (без заглядывания вперёд)."""
        if not isinstance(f, dict):
            return f
        keys = [k for k in f if k <= t]
        return f[max(keys)] if keys else None

    def _rows(self, ds, f, t_from, t_to, tau):
        t_list = np.arange(max(t_from, self.pre + 4), t_to, self.every)
        pot = potential(ds.liq, ds.up)
        bad = np.concatenate([np.zeros((1, ds.up.shape[1])), np.cumsum(ds.up < self.min_up, axis=0)])
        y = _changes(pot[:t_to], t_list, self.pre, FWD, bad=bad[:t_to + 1])            # [W, n, P]
        base = np.stack([np.nanmean(pot[t - self.pre:t], axis=0) for t in t_list])       # [n, P]
        scale = base + 20.0
        dc = _changes(lag_filter(ds.inj, tau, ds.step)[:t_to], t_list, self.pre, FWD)    # [W, n, I]
        dc = np.nan_to_num(dc)
        cols = [dc @ geo_weights(ds, self.radius)]
        has_f = np.ones(len(t_list), bool)
        if self.use_f:
            xf = np.zeros_like(cols[0])
            for n, t in enumerate(t_list):
                ft = self._f_at(f, t)
                if ft is None:
                    has_f[n] = False
                else:
                    xf[:, n] = dc[:, n] @ np.minimum(ft, self.f_cap)
            cols.append(xf)
        ys, xs = [], []
        for k in range(len(FWD)):
            yk = y[k] / scale
            ok = np.isfinite(yk)
            xk = [c[k] / scale for c in cols]
            # фиксированный эффект «участок x время»: вычитаем среднее по работавшим скважинам
            cnt = np.maximum(ok.sum(axis=1, keepdims=True), 1)
            yk = yk - np.where(ok, yk, 0).sum(axis=1, keepdims=True) / cnt
            xk = [x - np.where(ok, x, 0).sum(axis=1, keepdims=True) / cnt for x in xk]
            keep = ok & (ok.sum(axis=1, keepdims=True) >= 5) & has_f[:, None]
            ys.append(np.clip(yk[keep], -0.5, 0.5))
            xs.append(np.stack([x[keep] for x in xk], axis=1))
        return np.concatenate(ys), np.vstack(xs)

    # ---------- обучение ----------
    def fit(self, items):
        """items: список (ds, f, t_from, t_to) — участок, матрица связей CM и период обучения."""
        best = None
        self.scan_ = []
        for tau in self.taus:
            parts = [self._rows(ds, f, a, b, tau) for ds, f, a, b in items]
            y, X = np.concatenate([p[0] for p in parts]), np.vstack([p[1] for p in parts])
            coef, _ = nnls(X, y)
            sse = float(((y - X @ coef) ** 2).sum())
            self.scan_.append((tau, sse, coef))
            if best is None or sse < best[1]:
                best = (tau, sse, coef, float((y ** 2).sum()), len(y))
        self.tau, self.sse_, self.coef_, self.sst_, self.n_ = best
        self.g_geo = self.coef_[0]
        self.g_f = self.coef_[1] if self.use_f else 0.0
        return self

    def r2(self, items):
        """Доля объяснённой вариации изменений дебита на заданных периодах (может быть < 0)."""
        parts = [self._rows(ds, f, a, b, self.tau) for ds, f, a, b in items]
        y, X = np.concatenate([p[0] for p in parts]), np.vstack([p[1] for p in parts])
        return 1.0 - float(((y - X @ self.coef_) ** 2).sum()) / float((y ** 2).sum()), len(y)

    # ---------- применение ----------
    def gains(self, ds, f=None, t=None):
        """Матрица g[I,P]: м3/сут жидкости добывающей на 1 м3/сут закачки нагнетательной (установившийся отклик).
        f — связи CM (матрица или словарь по шагам; тогда t — момент, на который нужен прогноз)."""
        g = self.g_geo * geo_weights(ds, self.radius)
        ft = self._f_at(f, ds.T if t is None else t) if f is not None else None
        if self.use_f and ft is not None:
            g = g + self.g_f * np.minimum(ft, self.f_cap)
        return g

    def simulator(self, ds, f=None, t=None):
        """Функция (inj, up) -> отклик потенциального дебита, для сравнения сценариев закачки."""
        g = self.gains(ds, f, t)
        return lambda inj, up: lag_filter(inj, self.tau, ds.step) @ g

    def effect(self, ds, inj_base, inj_new, f=None, t=None):
        """Изменение дебита жидкости по добывающим [T,P] при замене плана закачки inj_base на inj_new."""
        sim = self.simulator(ds, f, t)
        return sim(inj_new, None) - sim(inj_base, None)
