"""Полусинтетика: реальные закачка, график работы и геометрия участка + заданный отклик.

Зачем: по фактическим данным истинные связи неизвестны, поэтому нельзя понять, плохо работает
метод или в данных просто нет сигнала. Здесь «истина» задаётся явно (матрица f, tau), дебиты
считаются ёмкостной моделью по реальной закачке, затем добавляется шум реального масштаба.
Если метод не восстанавливает заданный отклик даже здесь, дело в данных (мало изменений
закачки, большой шум), а не в модели.
"""
import dataclasses
import numpy as np
from .cm import CM
from .response import potential


def make_truth(ds, total_gain=(0.3, 0.9), tau=(30.0, 120.0), hetero=0.8, kappa=0.5, seed=0):
    """Истинная модель участка. total_gain — диапазон суммы f по добывающим для нагнетательной."""
    rng = np.random.default_rng(seed)
    n_inj, n_prod = ds.mask.shape
    w = np.where(ds.mask, 1.0 / np.maximum(ds.dist, 100.0), 0.0)
    w *= np.exp(hetero * rng.standard_normal(w.shape))            # неоднородность пласта
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    f = w * rng.uniform(*total_gain, size=(n_inj, 1))
    tau_j = rng.uniform(*tau, size=n_prod)
    pot = potential(ds.liq, ds.up)
    q_mean = np.nan_to_num(np.nanmean(pot, axis=0), nan=10.0)
    e = np.maximum(q_mean - ds.inj.mean(axis=0) @ f, 0.05 * q_mean)   # остальная добыча — не от закачки
    cm = CM(ds.mask, ds.dist, step=ds.step, tau_bounds=(3.0, 3000.0))
    cm.set_params(tau_j, e, q_mean, f, kappa, np.maximum(q_mean, 1.0))
    return cm


def make_dataset(ds, cm_true, rho=0.97, sigma_slow=0.03, sigma_fast=0.03, seed=0):
    """Синтетические дебиты: расчёт истинной модели x шум.

    Медленный шум (авторегрессия) имитирует изменения, не связанные с закачкой: при параметрах
    по умолчанию дебит за 26 недель «сам по себе» уходит на ~13 % (как в фактических данных).
    Быстрый шум — погрешность замеров."""
    rng = np.random.default_rng(seed + 1)
    q, _ = cm_true.simulate(ds.inj, ds.up)
    T, P = q.shape
    slow = np.zeros((T, P))
    slow[0] = rng.standard_normal(P) * sigma_slow / np.sqrt(1 - rho ** 2)
    for k in range(1, T):
        slow[k] = rho * slow[k - 1] + sigma_slow * rng.standard_normal(P)
    q_obs = q * np.exp(slow + sigma_fast * rng.standard_normal((T, P)))
    return dataclasses.replace(ds, liq=q_obs, oil=0.3 * q_obs)
