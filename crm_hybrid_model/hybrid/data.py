"""Подготовка данных: суточные замеры по скважинам -> регулярные ряды.

Вход — «длинная» таблица замеров (как BlockNN.csv из шахматки):
    Номер скважины; Дата; Характер работы (НЕФ/НАГ); Qж, м3/сут; Обводненность, %;
    Приемистость, м3/сут; ...
Запись в таблице есть только тогда, когда скважина работала и был замер.
Пропуск записей дольше `gap_prod`/`gap_inj` суток трактуется как остановка.

Выход — объект Dataset с рядами на регулярной сетке (шаг `step` суток):
    liq[T,P], oil[T,P]  — средние за шаг дебиты жидкости и нефти (с учётом простоев), м3/сут
    up[T,P]             — доля времени работы добывающей скважины за шаг (0..1)
    inj[T,I]            — средняя за шаг приёмистость, м3/сут
    mask[I,P]           — допустимые связи «нагнетательная -> добывающая»
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

COLS = {
    'Номер скважины': 'well', 'Дата': 'date', 'Характер работы': 'role',
    'Qж, м3/сут': 'ql', 'Обводненность, %': 'wc', 'Приемистость, м3/сут': 'inj',
    'Рзаб, атм': 'bhp', 'Qн, т/сут': 'qo_t',
}


@dataclass
class Dataset:
    dates: pd.DatetimeIndex      # начало каждого шага
    step: int                    # длина шага, сут
    prods: list
    injs: list
    liq: np.ndarray
    oil: np.ndarray
    up: np.ndarray
    inj: np.ndarray
    dist: np.ndarray             # [I,P], м
    mask: np.ndarray             # [I,P], bool
    pdist: np.ndarray            # [P,P], м
    bhp: np.ndarray = None       # [T,P], атм, NaN где замеров нет

    @property
    def T(self):
        return len(self.dates)

    def index_of(self, date):
        """Номер первого шага, начинающегося не раньше date."""
        return int(np.searchsorted(self.dates.values, np.datetime64(pd.Timestamp(date))))


def read_block(path, sep=';', encoding='cp1251'):
    df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
    df = df.rename(columns=COLS)
    df['well'] = df['well'].astype(str).str.strip()
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    df = df.dropna(subset=['date'])
    for c in ('ql', 'wc', 'inj', 'bhp'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def read_coords(path, encoding='cp1251'):
    c = pd.read_csv(path, sep=None, engine='python', encoding=encoding, index_col=0)
    c.index = c.index.astype(str).str.strip()
    c.columns = [str(x).strip().upper()[-1] for x in c.columns]   # 'Координата X' -> 'X'
    c = c[['X', 'Y']].apply(pd.to_numeric, errors='coerce').dropna()
    return c[~c.index.duplicated()]


def _daily(df, col, days, limit):
    """Суточная матрица [день, скважина]: протяжка замера вперёд не дольше limit суток."""
    piv = df.pivot_table(index='date', columns='well', values=col, aggfunc='mean')
    piv = piv.reindex(days)
    return piv.ffill(limit=limit)


def build_dataset(df, coords, step=7, gap_prod=14, gap_inj=5,
                  radius=1500.0, k_min=3, radius_max=3000.0, min_active_steps=26):
    days = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    pr = df[(df['role'] == 'НЕФ') & df['ql'].notna()].copy()
    pr['qo'] = pr['ql'] * (1.0 - pr['wc'].clip(0, 100) / 100.0)
    ir = df[(df['role'] == 'НАГ') & df['inj'].notna()]

    dl = _daily(pr, 'ql', days, gap_prod)
    do = _daily(pr, 'qo', days, gap_prod)
    di = _daily(ir, 'inj', days, gap_inj)
    db = _daily(pr[pr['bhp'] > 0], 'bhp', days, gap_prod).reindex(columns=dl.columns)
    act = dl.notna().astype(float)
    dl, do, di = dl.fillna(0.0), do.fillna(0.0), di.fillna(0.0)

    # агрегирование в шаги
    n = (len(days) // step) * step
    grp = np.arange(n) // step
    dates = days[:n:step]

    def agg(x):
        return x.iloc[:n].groupby(grp).mean().values

    liq, oil, up, inj = agg(dl), agg(do), agg(act), agg(di)
    bhp = db.iloc[:n].groupby(grp).mean().values          # среднее по дням с замером
    prods, injs = list(dl.columns), list(di.columns)

    # только скважины с координатами и достаточной историей
    kp = [j for j, w in enumerate(prods) if w in coords.index and (up[:, j] > 0).sum() >= min_active_steps]
    ki = [i for i, w in enumerate(injs) if w in coords.index and (inj[:, i] > 0).sum() >= min_active_steps]
    prods, injs = [prods[j] for j in kp], [injs[i] for i in ki]
    liq, oil, up, inj, bhp = liq[:, kp], oil[:, kp], up[:, kp], inj[:, ki], bhp[:, kp]

    xp, xi = coords.loc[prods].values, coords.loc[injs].values
    dist = np.linalg.norm(xi[:, None, :] - xp[None, :, :], axis=2)
    pdist = np.linalg.norm(xp[:, None, :] - xp[None, :, :], axis=2)
    mask = dist <= radius
    # каждой добывающей — не менее k_min ближайших нагнетательных (в пределах radius_max)
    order = np.argsort(dist, axis=0)
    for j in range(len(prods)):
        for i in order[:k_min, j]:
            if dist[i, j] <= radius_max:
                mask[i, j] = True
    mask &= dist > 1.0      # одна и та же скважина до/после перевода под закачку
    return Dataset(dates, step, prods, injs, liq, oil, up, inj, dist, mask, pdist, bhp)
