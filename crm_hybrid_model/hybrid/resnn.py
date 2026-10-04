"""Слой 3. Нейросетевая поправка: сколько «верить» физической модели.

Базовый прогноз — уровень на дату начала прогноза (origin), умноженный на график работы.
Физическая модель (CM) даёт ожидаемое изменение потенциального дебита относительно этого
уровня. Сеть по этим и другим признакам предсказывает фактическое изменение:

    q_hybrid(t) = [ level_j + qs_j * NN(x_j(t, origin)) ] * u_j(t)

Обучающая выборка строится «вложенно» и честно по времени: для каждой внутренней даты
начала прогноза CM обучается только на данных до неё, делает прогноз, и запоминается пара
«признаки -> фактическое изменение». Поэтому сеть видит реальную, а не подогнанную точность
CM и учится доверять ей ровно настолько, насколько она оправдывается.

Признаки используют только то, что известно на дату начала прогноза, плюс управляющие
воздействия прогнозного периода: закачку и график работы скважин (а при use_bhp=True —
ещё и заданное забойное давление, т. е. режим работы насоса).

Сеть — небольшой многослойный персептрон (scikit-learn), усреднение по нескольким запускам.
В том же классе есть линейный вариант (гребневая регрессия) для сравнения.
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .cm import CM

FEATURES = [
    'h',            # горизонт, доля от максимального
    'd_cm',         # изменение потенциального дебита по CM относительно старта
    'd_cm_anchor',  # то же по CM с привязкой состояния к факту
    'err0',         # ошибка свободного расчёта CM на старте
    'fit_err',      # ошибка настройки CM по скважине на истории
    'd_winj',       # изменение взвешенной закачки (сумма f_ij * I_i)
    'd_nb_up',      # изменение доли работы соседних добывающих
    'trend',        # наклон дебита за 8 шагов до старта
    'spread',       # разброс дебита за 8 шагов до старта
    'restart',      # шагов после последнего пуска (лог)
    'restart0',     # то же на дату старта
    'level',        # уровень на старте / средний дебит скважины
    'oil_cut',      # доля нефти на старте
    'tau', 'e', 'f_sum',
]
BHP_FEATURES = ['d_bhp', 'bhp_known']


def _since_restart(up):
    T, P = up.shape
    out = np.zeros((T, P))
    run = np.full(P, 50.0)
    for k in range(T):
        run = np.where(up[k] > 0.5, run + 1, 0.0)
        out[k] = run
    return out


def origin_block(ds, cm, t0, t1, n_neigh=4, horizon=26):
    """Признаки, цель и базовые прогнозы для одной даты начала прогноза."""
    qs = cm.qs
    inj_h, up_h, q_h = ds.inj[:t0], ds.up[:t0], ds.liq[:t0]
    inj_f, up_f = ds.inj[t0:t1], ds.up[t0:t1]
    n, P = up_f.shape
    _, s_hist = cm.simulate(inj_h, up_h)
    q_free, s_free = cm.forecast(inj_h, up_h, inj_f, up_f)
    q_anch, s_anch = cm.forecast(inj_h, up_h, inj_f, up_f, q_hist=q_h)

    last = slice(t0 - 4, t0)
    ok = up_h[last] > 0.5
    cnt = np.maximum(ok.sum(axis=0), 1)
    has = ok.sum(axis=0) >= 2
    level = np.where(ok, q_h[last] / np.maximum(up_h[last], 1e-6), 0).sum(axis=0) / cnt
    free0 = np.where(ok, s_hist[last], 0).sum(axis=0) / cnt
    level = np.where(has, level, s_hist[-1])
    free0 = np.where(has, free0, s_hist[-1])
    # масштаб нормировки: не меньше уровня, среднего дебита за историю и 5 м3/сут
    qs = np.maximum.reduce([level, cm.qs, np.full(P, 5.0)])
    hist = (up_h > 0.5).sum(axis=0)

    w8 = slice(max(t0 - 8, 0), t0)
    qa = np.where(up_h[w8] > 0.5, q_h[w8] / np.maximum(up_h[w8], 1e-6), np.nan)
    with np.errstate(all='ignore'):
        x = np.arange(qa.shape[0])[:, None] - (qa.shape[0] - 1) / 2
        m = np.nanmean(qa, axis=0)
        trend = np.nansum((qa - m) * x, axis=0) / np.maximum(np.nansum(np.isfinite(qa) * x ** 2, axis=0), 1e-9)
        spread = np.nanstd(qa, axis=0)
    trend, spread = np.nan_to_num(trend), np.nan_to_num(spread)

    winj = ds.inj[:t1] @ cm.f
    winj0 = winj[last].mean(axis=0)
    neigh = np.argsort(ds.pdist, axis=1)[:, 1:n_neigh + 1]
    nb_up = ds.up[:t1][:, neigh].mean(axis=2)
    nb0 = nb_up[last].mean(axis=0)
    oc = ds.oil[last].sum(axis=0) / np.maximum(q_h[last].sum(axis=0), 1e-9)
    restart = _since_restart(ds.up[:t1])

    bhp = ds.bhp[:t1]
    with np.errstate(all='ignore'):
        bhp0 = np.nanmean(bhp[last], axis=0)
    d_bhp = bhp[t0:t1] - bhp0
    bhp_known = np.isfinite(d_bhp).astype(float)
    d_bhp = np.nan_to_num(d_bhp) / 50.0

    one = np.ones((n, P))
    hh = np.repeat(np.arange(1, n + 1)[:, None], P, axis=1)
    cols = {
        'h': hh / horizon, 'd_cm': (s_free - free0) / qs, 'd_cm_anchor': (s_anch - level) / qs,
        'err0': one * (level - free0) / qs, 'fit_err': one * cm.fit_err_,
        'd_winj': (winj[t0:t1] - winj0) / qs, 'd_nb_up': nb_up[t0:t1] - nb0,
        'trend': one * trend / qs, 'spread': one * spread / qs,
        'restart': np.log1p(np.minimum(restart[t0:t1], 26.0)),
        'restart0': one * np.log1p(np.minimum(restart[t0 - 1], 26.0)),
        'level': one * level / qs, 'oil_cut': one * oc,
        'tau': one * cm.tau / 100.0, 'e': one * cm.e / qs, 'f_sum': one * cm.f.sum(axis=0),
        'd_bhp': d_bhp, 'bhp_known': bhp_known,
    }
    tab = pd.DataFrame({k: v.reshape(-1) for k, v in cols.items()})
    tab['origin'] = t0
    tab['k'] = np.repeat(np.arange(t0, t1), P)
    tab['j'] = np.tile(np.arange(P), n)
    tab['u'] = up_f.reshape(-1)
    tab['hist'] = np.tile(hist, n)
    q_fact = ds.liq[t0:t1]
    tab['y'] = ((q_fact / np.maximum(up_f, 0.5) - level) / qs).reshape(-1)
    base = {'level': level, 'qs': qs, 'cm_free': q_free, 'cm_anchor': q_anch,
            'cm_delta': np.maximum(level + s_free - free0, 0.0) * up_f, 'naive': level * up_f}
    return tab, base


def build_table(ds, origins, horizon=26, first=78, every=4, half_life=None, verbose=False):
    """Таблица «признаки -> цель» по сетке внутренних дат начала прогноза + заданным origins.

    Для каждой даты CM обучается только на данных до неё (тёплый старт от предыдущей).
    Возвращает таблицу, словарь базовых прогнозов и словарь обученных CM по датам.
    """
    grid = sorted(set(range(first, ds.T - 3, every)) | set(origins))
    tabs, bases, cms, theta = [], {}, {}, None
    for t0 in grid:
        cm = CM(ds.mask, ds.dist, step=ds.step)
        cm.fit(ds.inj[:t0], ds.up[:t0], ds.liq[:t0], theta0=theta, half_life=half_life,
               maxiter=2000 if theta is None else 400)
        theta = cm.theta
        tab, base = origin_block(ds, cm, t0, min(t0 + horizon, ds.T), horizon=horizon)
        tabs.append(tab)
        if t0 in origins:
            bases[t0], cms[t0] = base, cm
        if verbose:
            print('  origin', t0, 'loss', round(cm.loss_, 4), flush=True)
    return pd.concat(tabs, ignore_index=True), bases, cms


class Stacker:
    """Поправка к уровню: линейная ('ridge') или нейросетевая ('mlp')."""

    def __init__(self, kind='mlp', use_bhp=False, hidden=(16, 8), alpha=1.0, n_models=5,
                 max_iter=200, seed=0, min_hist=26, drop=()):
        self.min_hist = min_hist
        self.kind, self.hidden, self.alpha = kind, hidden, alpha
        self.n_models, self.max_iter, self.seed = n_models, max_iter, seed
        self.cols = [c for c in FEATURES + (BHP_FEATURES if use_bhp else []) if c not in drop]

    def fit(self, tab):
        tr = tab[(tab['u'] > 0.5) & (tab['hist'] >= self.min_hist)]
        X, y = tr[self.cols].values, np.clip(tr['y'].values, -2, 2)
        self.scaler = StandardScaler().fit(X)
        Xn = np.clip(self.scaler.transform(X), -5, 5)
        self.n_samples_ = len(y)
        if self.kind == 'ridge':
            self.models = [Ridge(alpha=100.0).fit(Xn, y)]
        else:
            self.models = [
                MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, max_iter=self.max_iter,
                             learning_rate_init=1e-3, random_state=self.seed + i, tol=1e-5,
                             batch_size=256).fit(Xn, y)
                for i in range(self.n_models)]
        return self

    def predict(self, tab):
        Xn = np.clip(self.scaler.transform(tab[self.cols].values), -5, 5)
        return np.clip(np.mean([m.predict(Xn) for m in self.models], axis=0), -2, 2)
