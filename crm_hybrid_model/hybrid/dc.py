"""Слой 2. Характеристики вытеснения (ХВ): доля нефти как функция накопленных отборов.

Все ХВ записаны в дифференциальной форме — как доля нефти fo в очередной порции жидкости:

    dVo = fo(состояние) * dVl

Так любую ХВ можно «прокрутить» вдоль любого прогноза жидкости (факт или расчёт CM),
и не нужно знать накопленные отборы до начала доступной истории: неизвестная предыстория
входит как подбираемый параметр сдвига (L0, W0) либо поглощается константой.

Библиотека (интегральная форма -> дифференциальная):
    Сазонов      Vo = A + B ln(Vl)        -> fo = B / (Vl + L0)
    Пирвердян    Vo = A - B / sqrt(Vl)    -> fo = B / (Vl + L0)^1.5
    Камбаров     Vo = A - B / Vl          -> fo = B / (Vl + L0)^2
    Экспонента   ln fo = a - b Vl         -> fo = f0 exp(-b Vl)
    ln(ВНФ)-Vo   ln WOR = a + b Vo        -> fo = 1 / (1 + exp(a + b Vo))
    Максимов     Vw = A exp(B Vo)         -> WOR = B (Vw + W0), fo = 1 / (1 + WOR)
    Постоянная   fo = const (база для сравнения)

Автоподбор: каждая ХВ настраивается на первых 80 % окна, проверяется на последних 20 %,
лучшие по ошибке накопленной нефти перенастраиваются на всём окне. Окно — вся история
или последние `recent` шагов (как в cost-БФА: «последние несколько лет»).
Прогноз всегда привязывается к последней фактической доле нефти (кривая проводится через
последние точки): без привязки ошибка на проверке в 1.3–2 раза выше.
"""
import numpy as np
from scipy.optimize import least_squares

EPS = 1e-9


def _power(m):
    def fo(p, vl, vo, vw):
        return p[0] / np.maximum(vl + p[1], EPS) ** m
    return fo


def _expo(p, vl, vo, vw):
    return p[0] * np.exp(-p[1] * vl)


def _lnwor(p, vl, vo, vw):
    return 1.0 / (1.0 + np.exp(np.clip(p[0] + p[1] * vo, -30, 30)))


def _maksimov(p, vl, vo, vw):
    return 1.0 / (1.0 + np.maximum(p[0] * (vw + p[1]), 0.0))


def _const(p, vl, vo, vw):
    return p[0] + 0.0 * vl


# имя: (функция, начальные параметры, нижние границы, верхние границы, нужна ли пошаговая прокрутка)
# объёмы внутри нормированы на накопленную жидкость окна, поэтому параметры порядка единицы
LIBRARY = {
    'Сазонов':    (_power(1.0), [0.3, 0.5], [1e-4, 1e-3], [50, 100], False),
    'Пирвердян':  (_power(1.5), [0.3, 0.5], [1e-4, 1e-3], [50, 100], False),
    'Камбаров':   (_power(2.0), [0.3, 0.5], [1e-4, 1e-3], [50, 100], False),
    'Экспонента': (_expo,       [0.5, 0.5], [1e-3, 0.0],  [1.0, 50], False),
    'ln(ВНФ)-Vo': (_lnwor,      [0.0, 1.0], [-20, 0.0],   [20, 200], True),
    'Максимов':   (_maksimov,   [1.0, 1.0], [1e-4, 0.0],  [500, 100], True),
    'Постоянная': (_const,      [0.5],      [0.0],        [1.0],     False),
}


def _roll(fo, p, dvl, explicit_ok, c=1.0, vl0=0.0, vo0=0.0, vw0=0.0):
    """Накопленная нефть по шагам при заданных приращениях жидкости dvl (нормированных)."""
    if explicit_ok:
        vl = vl0 + np.cumsum(dvl) - 0.5 * dvl
        f = np.clip(c * fo(p, vl, None, None), 0.0, 1.0)
        return vo0 + np.cumsum(f * dvl), f
    vl, vo, vw = vl0, vo0, vw0
    out, fs = np.empty(len(dvl)), np.empty(len(dvl))
    for k, d in enumerate(dvl):
        f = float(np.clip(c * fo(p, vl, vo, vw), 0.0, 1.0))
        vo += f * d
        vw += (1.0 - f) * d
        vl += d
        out[k], fs[k] = vo, f
    return out, fs


def _fit_one(name, dvl, dvo):
    fo, p0, lo, hi, stepwise = LIBRARY[name]
    target = np.cumsum(dvo)

    def resid(p):
        return _roll(fo, p, dvl, not stepwise)[0] - target

    best = None
    starts = [p0] if len(p0) == 1 else [p0, [p0[0] * 3, p0[1] * 0.2], [p0[0] * 0.3, p0[1] * 3]]
    if stepwise:
        starts = starts[:2]          # пошаговые кривые считаются дольше
    for s in starts:
        s = np.clip(s, lo, hi)
        try:
            r = least_squares(resid, s, bounds=(lo, hi), max_nfev=80 if stepwise else 200)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return None if best is None else best.x


def _anchor(fo, p, stepwise, dvl, dvo, n_last):
    """Множитель, проводящий кривую через фактическую долю нефти последних n_last шагов."""
    _, f = _roll(fo, p, dvl, not stepwise)
    sl = slice(-n_last, None)
    fact = dvo[sl].sum() / max(dvl[sl].sum(), EPS)
    model = (f[sl] * dvl[sl]).sum() / max(dvl[sl].sum(), EPS)
    return float(np.clip(fact / max(model, 1e-6), 0.3, 3.0))


class DisplacementCurve:
    """ХВ одной скважины (или элемента заводнения): автоподбор и прогноз доли нефти.

    mode='auto'      — одна лучшая кривая по проверке на последних 20 % окна;
    mode='ensemble'  — среднее по n_best лучшим кривым (устойчивее; разброс между кривыми
                       даёт сценарии P10/P50/P90, как в cost-БФА).
    """

    def __init__(self, mode='ensemble', n_best=3, recent=104, val_frac=0.2, anchor_steps=8, min_steps=30):
        self.mode, self.n_best = mode, n_best
        self.recent, self.val_frac = recent, val_frac
        self.anchor_steps, self.min_steps = anchor_steps, min_steps
        self.members = []          # (имя, окно, параметры, множитель привязки, масштаб, состояние)

    def fit(self, liq, oil, step=7.0):
        """liq, oil — ряды дебитов жидкости и нефти (м3/сут) на истории, с нулями при простоях."""
        act = liq > 0
        dvl_all, dvo_all = liq[act] * step, oil[act] * step
        self.n_act = len(dvl_all)
        self.members = []
        if self.n_act < self.min_steps:
            n = max(min(self.anchor_steps, self.n_act), 1)
            fo = float(dvo_all[-n:].sum() / max(dvl_all[-n:].sum(), EPS)) if self.n_act else 0.0
            self.members = [('Постоянная', 'короткая история', [fo], 1.0, 1.0, (0.0, 0.0, 0.0))]
            self.name, self.score_ = 'Постоянная', np.nan
            return self
        windows = [('вся история', self.n_act)]
        if self.n_act > self.recent:
            windows.append(('последние', self.recent))
        cands = []
        for wname, w in windows:
            dvl, dvo = dvl_all[-w:], dvo_all[-w:]
            scale = dvl.sum()
            dvl, dvo = dvl / scale, dvo / scale
            nv = max(int(len(dvl) * self.val_frac), 4)
            vl0, vo0 = dvl[:-nv].sum(), dvo[:-nv].sum()
            for name, (fo, _, _, _, stepwise) in LIBRARY.items():
                p = _fit_one(name, dvl[:-nv], dvo[:-nv])
                if p is None:
                    continue
                c = _anchor(fo, p, stepwise, dvl[:-nv], dvo[:-nv], self.anchor_steps)
                pred, _ = _roll(fo, p, dvl[-nv:], not stepwise, c, vl0, vo0, vl0 - vo0)
                err = abs(pred[-1] - dvo.sum()) / max(dvo[-nv:].sum(), EPS)
                cands.append((err, name, wname, w))
        cands.sort(key=lambda t: t[0])
        keep = cands[:1] if self.mode == 'auto' else cands[:self.n_best]
        for err, name, wname, w in keep:
            fo, _, _, _, stepwise = LIBRARY[name]
            dvl, dvo = dvl_all[-w:], dvo_all[-w:]
            scale = dvl.sum()
            dvl, dvo = dvl / scale, dvo / scale
            p = _fit_one(name, dvl, dvo)
            c = _anchor(fo, p, stepwise, dvl, dvo, self.anchor_steps)
            self.members.append((name, wname, p, c, scale, (dvl.sum(), dvo.sum(), dvl.sum() - dvo.sum())))
        self.name, self.score_ = keep[0][1], keep[0][0]
        return self

    def predict_members(self, liq_future, step=7.0):
        """Дебит нефти по каждой кривой ансамбля: массив [кривая, шаг]."""
        liq_future = np.asarray(liq_future, float)
        out = []
        for name, _, p, c, scale, (vl0, vo0, vw0) in self.members:
            fo, _, _, _, stepwise = LIBRARY[name]
            _, f = _roll(fo, p, liq_future * step / scale, not stepwise, c, vl0, vo0, vw0)
            out.append(f * liq_future)
        return np.array(out)

    def predict(self, liq_future, step=7.0):
        """Дебит нефти для прогнозного ряда дебита жидкости (среднее по ансамблю)."""
        return self.predict_members(liq_future, step).mean(axis=0)
