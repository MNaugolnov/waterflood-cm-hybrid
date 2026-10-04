"""Проверка на синтетике: модели должны восстанавливать заданные параметры.

Запуск:  python tests/test_synthetic.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid.cm import CM
from hybrid.dc import DisplacementCurve
from hybrid.data import Dataset
from hybrid.synth import make_truth, make_dataset
from hybrid.response import find_events, observed_response, model_response
from hybrid.respmodel import ResponseModel


def test_cm_recovers_connectivity():
    rng = np.random.default_rng(1)
    T, I, P = 250, 4, 5
    f_true = rng.dirichlet(np.ones(P + 1), size=I)[:, :P]          # сумма по строке < 1
    tau_true = rng.uniform(15, 90, P)
    # закачка — независимые ступеньки по скважинам (сигнал, по которому связи различимы)
    inj = np.repeat(rng.uniform(80, 320, (T // 10, I)), 10, axis=0)
    up = np.ones((T, P))
    up[100:110, 2] = 0.0                                           # остановка одной скважины
    mask = np.ones((I, P), bool)
    gen = CM(mask, np.full((I, P), 500.0))
    gen.qs = np.full(P, 100.0)
    a = np.log((tau_true - gen.tau_lo) / (gen.tau_hi - tau_true))
    z = np.log(f_true / (1 - f_true.sum(axis=1, keepdims=True)))
    gen.theta = np.concatenate([a, np.full(P, -20.0), np.full(P, 0.5), z.ravel(), [0.0]])
    q, _ = gen.simulate(inj, up)
    q_noisy = q * (1 + 0.03 * rng.standard_normal(q.shape))
    m = CM(mask, np.full((I, P), 500.0)).fit(inj, up, q_noisy, maxiter=2000)
    err_f = np.abs(m.f - f_true).max()
    mae_f = np.abs(m.f - f_true).mean()
    err_tau = np.abs(m.tau / tau_true - 1).max()
    print(f'CM: ошибка f средняя = {mae_f:.3f}, макс. = {err_f:.3f}; макс. отн. ошибка tau = {err_tau:.2f}')
    assert err_f < 0.05 and err_tau < 0.25


def test_dc_follows_water_cut_growth():
    step, n = 7.0, 200
    liq = np.full(n, 100.0)
    vl = np.cumsum(liq * step)
    oil_cut = 0.9 / (1 + vl / 40000.0) ** 2                        # Камбаров в дифференциальной форме
    oil = liq * oil_cut
    dc = DisplacementCurve().fit(liq[:150], oil[:150], step)
    pred = dc.predict(liq[150:], step)
    err = abs(pred.sum() / oil[150:].sum() - 1)
    const = abs(oil[142:150].mean() * 50 / oil[150:].sum() - 1)
    print(f'ХВ: ошибка накопленной нефти = {100 * err:.1f} % (постоянная доля нефти: {100 * const:.1f} %), '
          f'кривые: {[m[0] for m in dc.members]}')
    assert err < 0.05 and err < const


def _grid_dataset(seed=3, T=230):
    """Искусственный участок: 9 нагнетательных и 16 добывающих на сетке 500 м, закачка ступеньками."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    xi = np.array([[x, y] for x in (500, 1500, 2500) for y in (500, 1500, 2500)], float)
    xp = np.array([[x, y] for x in (0, 1000, 2000, 3000) for y in (0, 1000, 2000, 3000)], float)
    dist = np.linalg.norm(xi[:, None] - xp[None], axis=2)
    pdist = np.linalg.norm(xp[:, None] - xp[None], axis=2)
    inj = np.zeros((T, 9))
    for i in range(9):                       # у каждой скважины свои моменты смены режима
        k = 0
        while k < T:
            n = int(rng.integers(10, 30))
            inj[k:k + n, i] = rng.uniform(100, 400)
            k += n
    up = np.ones((T, 16))
    liq = np.full((T, 16), 150.0)
    return Dataset(pd.date_range('2015-01-01', periods=T, freq='7D'), 7, [f'p{k}' for k in range(16)],
                   [f'i{k}' for k in range(9)], liq, 0.3 * liq, up, inj, dist, dist <= 1200, pdist,
                   np.full((T, 16), np.nan))


def test_response_tools_recover_gain():
    ds0 = _grid_dataset()
    truth = make_truth(ds0, total_gain=(0.4, 0.8), seed=5)
    ds = make_dataset(ds0, truth, seed=5, sigma_slow=0.005, sigma_fast=0.01)      # слабый шум: проверяем сам расчёт
    ev = find_events(ds)
    obs = observed_response(ds, ev, ctrl_radius=1800.0)
    tru = model_response(lambda inj, up: truth.simulate(inj, up)[1], ds, ev)
    rm = ResponseModel(use_f=False).fit([(ds, None, 12, ds.T)])
    mod = model_response(rm.simulator(ds), ds, ev)
    t = obs.merge(tru, on=['ev', 'j']).merge(mod, on=['ev', 'j'], suffixes=(' истина', ' модель'))
    n = t[t.dist <= 1000].dropna(subset=['17-26 нед'])
    sgn, den = np.sign(n.d), n.d.abs().sum()
    r_obs = (n['17-26 нед'] * sgn).sum() / den
    r_true = (n['мод 17-26 нед истина'] * sgn).sum() / den
    r_mod = (n['мод 17-26 нед модель'] * sgn).sum() / den
    print(f'Отклик: истина {r_true:.3f}, событийная оценка {r_obs:.3f}, модель отклика {r_mod:.3f} '
          f'(событий {n.ev.nunique()}, tau модели {rm.tau:.0f} сут)')
    assert abs(r_obs / r_true - 1) < 0.3 and abs(r_mod / r_true - 1) < 0.3


if __name__ == '__main__':
    test_cm_recovers_connectivity()
    test_dc_follows_water_cut_growth()
    test_response_tools_recover_gain()
    print('OK')
