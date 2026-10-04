"""Проверка расчётного ядра на данных из sample_data (Dash для неё не нужен).

Запуск из папки crm_webapp:   python tests/test_core.py
"""
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings('ignore')

import config                                  # noqa: E402
from core import engine, figures, reader       # noqa: E402

D = os.path.join(config.HERE, 'sample_data')


def raw(name):
    return reader.read_path(os.path.join(D, name))


def test_reader_formats():
    prod, inj, xy = raw('production.csv'), raw('injection.csv'), raw('coordinates.csv')
    df, coords, info = reader.assemble(prod, inj, xy)
    assert info['wells_prod'] > 10 and info['wells_inj'] > 5 and info['has_oil']
    assert reader.detect_step(df[df.role == 'НЕФ'])[0] == 7

    # русские заголовки, запятая как разделитель, десятичная запятая, Windows-1251
    ru = prod.rename(columns={'Well_ID': 'Скважина', 'Date': 'Дата', 'Qliq': 'Дебит жидкости, м3/сут',
                              'WC': 'Обводненность, %', 'BHP': 'Рзаб, атм'})
    ru['Дебит жидкости, м3/сут'] = ru['Дебит жидкости, м3/сут'].str.replace('.', ',', regex=False)
    buf = ru.to_csv(sep='\t', index=False).encode('cp1251')
    p2, _ = reader.parse_production(reader.read_table(buf, 'добыча.txt'))
    p1, _ = reader.parse_production(prod)
    assert np.allclose(p1['ql'].fillna(-1), p2['ql'].fillna(-1)) and (p1['date'].values == p2['date'].values).all()

    # дебит нефти вместо обводнённости
    q = prod.copy()
    q['Qoil'] = (pd.to_numeric(q['Qliq']) * (1 - pd.to_numeric(q['WC']) / 100)).astype(str)
    p3, _ = reader.parse_production(q.drop(columns='WC'))
    ok = p1['ql'] > 0
    assert np.allclose(p1.loc[ok, 'wc'], p3.loc[ok, 'wc'], atol=1e-6)

    # «широкая» таблица закачки и даты в формате ГГГГ-ММ-ДД
    long = reader.parse_injection(inj)
    wide = long.pivot_table(index='date', columns='well', values='inj').reset_index()
    wide['date'] = wide['date'].dt.strftime('%Y-%m-%d')
    w2 = reader.parse_injection(wide.astype(str).replace('nan', np.nan))
    ref = long.groupby(['well', 'date'])['inj'].mean().sum()          # в исходных данных есть повторные записи
    assert abs(w2['inj'].sum() - ref) < 1e-6 * ref

    # Excel
    bio = io.BytesIO()
    xy.to_excel(bio, index=False)
    c2 = reader.parse_coords(reader.read_table(bio.getvalue(), 'coords.xlsx'))
    assert c2.equals(coords)

    # закачка в одном файле с добычей (как BlockN.csv)
    both = pd.concat([prod.assign(role='НЕФ'), inj.rename(columns={'Qinj': 'Приемистость'}).assign(role='НАГ')])
    both = both.rename(columns={'role': 'Характер работы'})
    df2, _, _ = reader.assemble(both, None, xy)
    assert len(df2) == len(df)

    # понятные сообщения об ошибках
    for bad, part in ((prod.drop(columns='Qliq'), 'дебит жидкости'), (prod.drop(columns='Date'), 'дата')):
        try:
            reader.assemble(bad, inj, xy)
            raise AssertionError('ожидалась ошибка')
        except reader.DataError as e:
            assert part in str(e)
    try:
        reader.assemble(prod, None, xy)
        raise AssertionError('ожидалась ошибка')
    except reader.DataError as e:
        assert 'закачки' in str(e)


def test_pipeline():
    m = engine.load_files(raw('production.csv'), raw('injection.csv'), raw('coordinates.csv'), 'sample')
    ds = m.ds
    assert ds.step == 7 and len(ds.prods) >= 10 and len(ds.injs) >= 5
    assert m.info['cm_wmape'] < 0.3
    assert m.g.shape == (len(ds.injs), len(ds.prods)) and (m.g >= 0).all() and m.g.sum(axis=1).max() < 0.5

    pairs = engine.pairs_table(m, max_dist=1000)
    assert len(pairs) and (pairs['dist'] <= 1000).all() and pairs['coef'].is_monotonic_decreasing
    figures.well_map(m, pairs, [ds.prods[0]])
    figures.history(m, [])

    fc = engine.forecast(m, ds.T - 30, 26, with_oil=False)
    mt = fc['metrics']
    assert fc['n_fact'] == 26 and float(mt[mt.method == 'naive'].block_wmape.iloc[0]) < 0.3
    fw = engine.forecast(m, ds.T, 13, with_oil=False)
    assert fw['n_fact'] == 0 and fw['metrics'] is None and np.isfinite(fw['pred']['cm']).all()
    figures.forecast(m, fc, None, 'liq')
    figures.forecast(m, fw, ds.prods[0], 'liq')

    es = engine.event_study(m)
    assert es['n_events'] > 10 and len(es['summary']) == 3
    figures.response(es['summary'])

    # сценарий: нулевое изменение — нулевой эффект; рост закачки — рост жидкости; подбор сохраняет сумму
    z = engine.scenario(m, np.zeros(len(ds.injs)), 26)
    assert abs(z['liq_end']) < 1e-9 and abs(z['cm_end']) < 1e-6
    i = int(np.argmax(m.inj_now))
    d = np.zeros(len(ds.injs))
    d[i] = 100.0
    s = engine.scenario(m, d, 26)
    steady = float((m.g[i] * m.up_now).sum() * 100.0)
    assert 0 <= s['liq_end'] <= steady + 1e-9 and s['liq_end'] > 0.8 * steady          # за полгода при tau = 60 сут
    assert (np.diff(s['d_liq'].sum(axis=1)) >= -1e-9).all()
    a = engine.suggest_delta(m, 0.2, 'oil')
    assert abs(a.sum()) < 1e-6 and (np.abs(a) <= 0.2 * m.inj_now + 1e-9).all()
    assert engine.scenario(m, a, 26)['oil_end'] >= -1e-9
    mv = engine.manual_delta(m, [ds.injs[i]], [], 50)
    assert abs(mv.sum() + 50) < 1e-9
    figures.scenario_time(s)
    figures.scenario_wells(m, s)


def test_monthly():
    """Те же данные, сведённые к месячным: шаг определяется как 30 сут, расчёт проходит."""
    prod, inj = raw('production.csv'), raw('injection.csv')
    p, _ = reader.parse_production(prod)
    p['month'] = p['date'].dt.to_period('M').dt.to_timestamp()
    days = p['month'].dt.days_in_month
    p['oil'] = p['ql'] * (1 - p['wc'] / 100)
    pm = p.groupby(['well', 'month']).agg(ql=('ql', 'mean'), oil=('oil', 'mean')).reset_index()
    pm['wc'] = 100 * (1 - pm['oil'] / pm['ql'].where(pm['ql'] > 0))
    pm = pm.rename(columns={'well': 'Скважина', 'month': 'Дата', 'ql': 'Qж', 'wc': 'Обводненность'})
    i = reader.parse_injection(inj)
    i['month'] = i['date'].dt.to_period('M').dt.to_timestamp()
    im = i.groupby(['well', 'month'])['inj'].mean().reset_index()
    im.columns = ['Скважина', 'Дата', 'Приемистость']
    m = engine.load_files(pm[['Скважина', 'Дата', 'Qж', 'Обводненность']].astype(str), im.astype(str),
                          raw('coordinates.csv'), 'monthly')
    assert m.ds.step == 30 and m.ds.T > 40
    fc = engine.forecast(m, m.ds.T - 6, 6, with_oil=False)
    assert fc['n_fact'] == 6
    es = engine.event_study(m)
    assert es['windows'] == ['1-2 мес', '3-4 мес', '5-6 мес']
    engine.scenario(m, engine.suggest_delta(m), 6)


if __name__ == '__main__':
    for f in (test_reader_formats, test_pipeline, test_monthly):
        f()
        print('ok ', f.__name__)
    print('Все проверки пройдены.')
