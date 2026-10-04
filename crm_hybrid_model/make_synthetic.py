"""Синтетический участок заводнения для примеров и тестов.

Реальные промысловые данные в репозиторий не входят. Этот сценарий создаёт участок с известной
«истиной»: скважины на регулярной сетке, закачка со ступенями и остановами, дебиты жидкости по
ёмкостной модели с заданными связями, шум, рост обводнённости, редкие замеры.

    python make_synthetic.py

Записывает одни и те же данные в двух форматах:
    data/Block S1/BlockS1.csv, coord.csv        — формат выгрузки (один файл, столбец «Характер работы»);
    ../crm_webapp/sample_data/*.csv             — три отдельных файла для веб-приложения;
    data/Block S1/truth_f.csv, truth_wells.csv  — заданные связи f и параметры скважин (для сравнения).
"""
import argparse
import os
import numpy as np
import pandas as pd
from hybrid.cm import CM

HERE = os.path.dirname(os.path.abspath(__file__))


def geometry(rng, rows=5, cols=7, spacing=450.0):
    """Обращённая девятиточечная сетка: нагнетательные в узлах с нечётными номерами ряда и столбца."""
    prods, injs = [], []
    for r in range(rows):
        for c in range(cols):
            xy = (10000.0 + c * spacing + rng.normal(0, 30), 20000.0 + r * spacing + rng.normal(0, 30))
            (injs if r % 2 == 1 and c % 2 == 1 else prods).append(xy)
    return np.array(prods), np.array(injs)


def injection(rng, T, n_inj):
    """Приёмистость по суткам: кусочно-постоянный режим со ступенями и остановами."""
    inj = np.zeros((T, n_inj))
    for i in range(n_inj):
        base = rng.uniform(250, 600)
        t = int(rng.integers(250, 500)) if i == n_inj - 1 else 0        # одна скважина введена позже
        while t < T:
            if rng.random() < 0.12:
                n, level = int(rng.integers(25, 80)), 0.0                # останов
            else:
                n, level = int(rng.integers(60, 240)), base * rng.uniform(0.5, 1.5)
            inj[t:t + n, i] = level
            t += n
    return inj * np.exp(0.03 * rng.standard_normal(inj.shape))


def uptime(rng, T, n_prod):
    """Работа добывающих по суткам: остановки на 3-9 недель, поздний ввод, выбытие."""
    up = np.ones((T, n_prod))
    for j in range(n_prod):
        for _ in range(rng.poisson(4)):
            t = int(rng.integers(0, T))
            up[t:t + int(rng.integers(20, 65)), j] = 0.0
    late = rng.choice(n_prod, 3, replace=False)
    for j in late:
        up[:int(rng.integers(200, 500)), j] = 0.0
    gone = rng.choice([j for j in range(n_prod) if j not in late], 2, replace=False)
    for j in gone:
        up[int(rng.integers(1300, 1600)):, j] = 0.0
    return up


def make(seed=7, start='2015-01-01', end='2019-09-30', total_gain=(0.25, 0.6), tau=(30.0, 120.0)):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq='D')
    T = len(days)
    xp, xi = geometry(rng)
    P, n_inj = len(xp), len(xi)
    prods = [str(101 + j) for j in range(P)]
    injs = [str(201 + i) for i in range(n_inj)]
    dist = np.linalg.norm(xi[:, None, :] - xp[None, :, :], axis=2)
    mask = dist <= 1000.0
    inj, up = injection(rng, T, n_inj), uptime(rng, T, P)

    # истинная ёмкостная модель
    w = np.where(mask, 1.0 / np.maximum(dist, 100.0), 0.0) * np.exp(0.8 * rng.standard_normal(dist.shape))
    f = w / w.sum(axis=1, keepdims=True) * rng.uniform(*total_gain, size=(n_inj, 1))
    tau_j = rng.uniform(*tau, size=P)
    q_mean = np.exp(rng.normal(np.log(80.0), 0.5, P)).clip(25, 300)
    e = np.maximum(q_mean - inj.mean(axis=0) @ f, 0.2 * q_mean)        # приток, не связанный с закачкой
    cm = CM(mask, dist, step=1.0, tau_bounds=(3.0, 3000.0))
    cm.set_params(tau_j, e, q_mean, f, 0.5, np.maximum(q_mean, 1.0))
    q, _ = cm.simulate(inj, up)

    # шум: медленный (изменения, не связанные с закачкой) и быстрый (погрешность замера)
    rho, sd = 0.97 ** (1 / 7), 0.12
    slow = np.zeros((T, P))
    slow[0] = sd * rng.standard_normal(P)
    for k in range(1, T):
        slow[k] = rho * slow[k - 1] + sd * np.sqrt(1 - rho ** 2) * rng.standard_normal(P)
    q = q * np.exp(slow + 0.04 * rng.standard_normal((T, P)))

    # обводнённость: водонефтяной фактор растёт с накопленным отбором жидкости
    wc0, wc1 = rng.uniform(0.2, 0.6, P), rng.uniform(0.75, 0.95, P)
    wor0, wor1 = wc0 / (1 - wc0), wc1 / (1 - wc1)
    vl = np.cumsum(q, axis=0)
    wor = wor0 * np.exp(np.log(wor1 / wor0) * vl / vl[-1])
    wc = (100 * wor / (1 + wor) + 1.5 * rng.standard_normal((T, P))).clip(0, 99)
    bhp = rng.uniform(40, 90, P) + 3 * rng.standard_normal((T, P))

    # записи замеров: по добывающим — в среднем раз в 2-3 дня, по нагнетательным — ежедневно
    rows = []
    for j, well in enumerate(prods):
        last = -10
        for k in np.where(up[:, j] > 0)[0]:
            if rng.random() < 0.4 or k - last >= 5:
                last = k
                rows.append((well, days[k], 'НЕФ', round(q[k, j], 1), int(round(wc[k, j])), round(bhp[k, j]), np.nan))
    for i, well in enumerate(injs):
        for k in np.where(inj[:, i] > 1.0)[0]:
            rows.append((well, days[k], 'НАГ', np.nan, np.nan, np.nan, round(inj[k, i])))
    rec = pd.DataFrame(rows, columns=['well', 'date', 'role', 'ql', 'wc', 'bhp', 'inj']).sort_values(['date', 'well'])
    coords = pd.DataFrame(np.vstack([xp, xi]).round(0), index=prods + injs, columns=['X', 'Y']).astype(int)
    truth_f = pd.DataFrame(f, index=injs, columns=prods).round(4)
    truth_w = pd.DataFrame({'tau, сут': tau_j, 'e, м3/сут': e}, index=prods).round(2)
    return rec, coords, truth_f, truth_w


def write(rec, coords, truth_f, truth_w, model_dir, web_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(web_dir, exist_ok=True)
    d = rec['date'].dt.strftime('%d.%m.%Y')
    block = pd.DataFrame({
        'Номер скважины': rec['well'], 'Дата': d, 'Характер работы': rec['role'], 'Пласт': 'S1',
        'Обводненность, %': rec['wc'], 'Qж, м3/сут': rec['ql'], 'Рзаб, атм': rec['bhp'],
        'Приемистость, м3/сут': rec['inj']})
    block.to_csv(os.path.join(model_dir, 'BlockS1.csv'), sep=';', index=False, encoding='cp1251')
    coords.rename_axis('Скважина').to_csv(os.path.join(model_dir, 'coord.csv'), sep=';', encoding='cp1251')
    truth_f.to_csv(os.path.join(model_dir, 'truth_f.csv'), sep=';', encoding='utf-8-sig')
    truth_w.to_csv(os.path.join(model_dir, 'truth_wells.csv'), sep=';', encoding='utf-8-sig')
    p = rec[rec.role == 'НЕФ']
    pd.DataFrame({'Well_ID': p['well'], 'Date': d[p.index], 'Qliq': p['ql'], 'WC': p['wc'].astype(int),
                  'BHP': p['bhp'].astype(int)}).to_csv(os.path.join(web_dir, 'production.csv'), sep=';', index=False,
                                                       encoding='utf-8-sig')
    i = rec[rec.role == 'НАГ']
    pd.DataFrame({'Well_ID': i['well'], 'Date': d[i.index], 'Qinj': i['inj'].astype(int)}).to_csv(
        os.path.join(web_dir, 'injection.csv'), sep=';', index=False, encoding='utf-8-sig')
    coords.rename_axis('Well_ID').to_csv(os.path.join(web_dir, 'coordinates.csv'), sep=';', encoding='utf-8-sig')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out-model', default=os.path.join(HERE, 'data', 'Block S1'))
    ap.add_argument('--out-web', default=os.path.join(HERE, '..', 'crm_webapp', 'sample_data'))
    a = ap.parse_args()
    out = make(a.seed)
    write(*out, a.out_model, a.out_web)
    rec = out[0]
    print(f'Записано: {rec[rec.role == "НЕФ"].well.nunique()} добывающих, {rec[rec.role == "НАГ"].well.nunique()} '
          f'нагнетательных, {len(rec)} записей, {rec.date.min().date()} - {rec.date.max().date()}')
