"""Запуск гибридной модели CM + ХВ + нейросеть на участках (блоках) месторождения.

Пример:
    python run_hybrid.py --data data --blocks S1 --out results

Для каждого участка нужны два файла в папке "<data>/Block N":
    BlockN.csv   — суточные замеры (Номер скважины; Дата; Характер работы; Qж; Обводненность; Приемистость)
    coord*.csv   — координаты скважин (индекс — номер скважины, столбцы X и Y)

Зависимости: numpy, scipy, pandas, scikit-learn.
"""
import argparse
import glob
import os
import time
import warnings
import pandas as pd
from hybrid.data import read_block, read_coords, build_dataset
from hybrid.evaluate import prepare, evaluate, summarize, LIQ, OIL

warnings.filterwarnings('ignore')


def load_blocks(data_dir, names, step):
    blocks = {}
    for b in names:
        d = os.path.join(data_dir, f'Block {b}')
        fb = [f for f in glob.glob(os.path.join(d, '*')) if os.path.basename(f).lower() == f'block{b}.csv'.lower()]
        fc = sorted(glob.glob(os.path.join(d, 'coord*.csv')))
        if not fb or not fc:
            print(f'участок {b}: не найдены файлы данных или координат, пропуск')
            continue
        blocks[b] = build_dataset(read_block(fb[0]), read_coords(fc[0]), step=step)
    if not blocks:
        raise SystemExit(f'В папке "{data_dir}" не найдено ни одного участка из списка: {", ".join(names)}')
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='папка с подпапками "Block N"')
    ap.add_argument('--blocks', nargs='+', default=['S1'])
    ap.add_argument('--origins', nargs='+', default=['2018-01-01', '2018-07-01', '2019-01-01'],
                    help='даты начала прогноза')
    ap.add_argument('--horizon', type=int, default=26, help='горизонт прогноза, шагов')
    ap.add_argument('--step', type=int, default=7, help='шаг по времени, сут')
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-oil', action='store_true')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--save-prep', action='store_true', help='сохранить подготовленные таблицы (prep.pkl)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = time.time()

    print('Чтение данных...')
    blocks = load_blocks(a.data, a.blocks, a.step)
    print('Настройка CM по сетке дат и сбор обучающей таблицы...')
    prep = prepare(blocks, a.origins, horizon=a.horizon)
    if a.save_prep:
        import pickle
        with open(os.path.join(a.out, 'prep.pkl'), 'wb') as fh:
            pickle.dump(prep, fh)
    print('Обучение поправки и прогноз...')
    df, fc = evaluate(prep, range(len(a.origins)), horizon=a.horizon, with_oil=not a.no_oil, n_jobs=a.jobs)

    df.to_csv(os.path.join(a.out, 'metrics_by_well.csv'), index=False, encoding='utf-8-sig')
    cols = LIQ + ([] if a.no_oil else OIL)
    s = summarize(df, cols)
    s.round(2).to_csv(os.path.join(a.out, 'summary.csv'), encoding='utf-8-sig')
    by_block = df[df.eligible].groupby('block')[cols].median().mul(100).round(1)
    by_block.to_csv(os.path.join(a.out, 'summary_by_block.csv'), encoding='utf-8-sig')

    last = len(a.origins) - 1
    for b, c in prep.items():
        ds, cm = c['ds'], c['cms'][c['origins'][last]]
        pd.DataFrame(cm.f, index=ds.injs, columns=ds.prods).round(3).to_csv(
            os.path.join(a.out, f'connectivity_block{b}.csv'), encoding='utf-8-sig')
        pd.DataFrame({'tau, сут': cm.tau, 'e, м3/сут': cm.e, 'ошибка настройки CM': cm.fit_err_},
                     index=ds.prods).round(3).to_csv(os.path.join(a.out, f'well_params_block{b}.csv'),
                                                     encoding='utf-8-sig')
        f = fc[(b, last)]
        rows = []
        for k, v in [('факт_жидкость', f['liq']), ('факт_нефть', f['oil'])] + list(f['pred'].items()):
            x = pd.DataFrame(v, index=f['dates'], columns=ds.prods).stack().rename(k)
            rows.append(x)
        pd.concat(rows, axis=1).round(2).rename_axis(['дата', 'скважина']).to_csv(
            os.path.join(a.out, f'forecast_block{b}.csv'), encoding='utf-8-sig')

    pd.set_option('display.width', 200)
    print(f'\nСкважино-прогнозов в сравнении: {int(df.eligible.sum())}')
    print(s.round(2).to_string())
    print('\nМедиана wMAPE по участкам, %:')
    print(by_block.to_string())
    print(f'\nВремя расчёта: {time.time() - t:.0f} с. Результаты: {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
