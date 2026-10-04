"""Чтение исходных данных: добыча, закачка, координаты скважин.

Принимаются файлы CSV/TXT (разделитель и кодировка определяются сами) и Excel.
Названия столбцов распознаются по словарю синонимов (русские и английские), порядок любой.

    Добыча      — скважина, дата, дебит жидкости (м3/сут) и обводнённость (%) либо дебит нефти;
                  по желанию забойное давление.
    Закачка     — скважина, дата, приёмистость (м3/сут). Допустима и «широкая» таблица:
                  первый столбец — дата, остальные — скважины.
    Координаты  — скважина, X, Y (м).

Если закачка лежит в том же файле, что и добыча (как в выгрузке BlockN.csv: столбцы
«Характер работы» и «Приемистость»), файл закачки можно не задавать.

Результат — таблица в том виде, который ждёт hybrid.data.build_dataset:
    well, date, role ('НЕФ' / 'НАГ'), ql, wc, inj, bhp
"""
import base64
import io
import re
import numpy as np
import pandas as pd

# (точные имена, подстроки) — в порядке приоритета; всё в нижнем регистре
ALIASES = {
    'well': (['well_id', 'well', 'скважина', 'номер скважины', '№ скважины', 'скв', 'uwi', 'wellname', 'well_name'],
             ['скваж', 'well']),
    'date': (['date', 'дата', 'time', 'месяц', 'период'], ['дата', 'date']),
    'ql':   (['qliq', 'ql', 'qж', 'q_liq', 'liq', 'liquid_rate', 'дебит жидкости'], ['qж', 'qliq', 'жидкост', 'liq']),
    'qo':   (['qoil', 'qo', 'qн', 'q_oil', 'oil_rate', 'дебит нефти'], ['qн', 'qoil', 'нефт', 'oil']),
    'wc':   (['%', 'wc', 'wct', 'watercut', 'water_cut', 'обводненность', 'обводнённость'],
             ['обводн', 'watercut', 'water cut', 'wct']),
    'inj':  (['inj', 'qinj', 'winj', 'q_inj', 'приемистость', 'приёмистость', 'закачка'],
             ['приемист', 'приёмист', 'закачк', 'qinj', 'inj']),
    'bhp':  (['bhp', 'pwf', 'рзаб', 'pзаб'], ['рзаб', 'забойн', 'bhp', 'pwf']),
    'role': (['role', 'характер работы', 'назначение', 'тип скважины', 'well_type'], ['характер']),
}


class DataError(ValueError):
    """Ошибка в исходных данных — текст показывается пользователю."""


# ---------- файл -> таблица строк ----------
def decode_upload(contents):
    """Содержимое dcc.Upload ('data:...;base64,XXXX') -> bytes."""
    return base64.b64decode(contents.split(',', 1)[1])


def read_table(raw, filename=''):
    """bytes -> DataFrame, все значения строками."""
    name = (filename or '').lower()
    if name.endswith(('.xlsx', '.xlsm', '.xls')):
        return pd.read_excel(io.BytesIO(raw), dtype=str)
    text = None
    for enc in ('utf-8-sig', 'cp1251'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DataError(f'{filename}: не удалось определить кодировку (ожидается UTF-8 или Windows-1251)')
    head = text[:5000].split('\n')[0]
    sep = max((';', '\t', ',', '|'), key=head.count)
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, engine='python')
    return df.mask(df == '')                      # пустые ячейки -> пропуски


def read_path(path):
    with open(path, 'rb') as fh:
        return read_table(fh.read(), path)


# ---------- распознавание столбцов ----------
def _norm(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def find_columns(df, keys):
    """{ключ: имя столбца} для найденных ключей; сначала точные совпадения, затем по подстроке."""
    names = {c: _norm(c) for c in df.columns}
    found, used = {}, set()
    for exact in (True, False):
        for k in keys:
            if k in found:
                continue
            for a in ALIASES[k][0 if exact else 1]:
                hit = [c for c, n in names.items() if c not in used and
                       (n == a or n.split(',')[0].strip() == a if exact else a in n)]
                if hit:
                    found[k] = hit[0]
                    used.add(hit[0])
                    break
    return found


def to_number(s):
    s = s.astype(str).str.replace(' ', '', regex=False).str.replace(' ', '', regex=False)
    return pd.to_numeric(s.str.replace(',', '.', regex=False), errors='coerce')


def to_date(s):
    """Даты в форматах 31.12.2019, 2019-12-31, 12/31/2019, 31/12/2019, а также даты Excel."""
    s = s.astype(str).str.strip()
    probe = s[s.str.len() > 0].head(2000)
    if probe.str.match(r'^\d{4}-\d{1,2}-\d{1,2}').mean() > 0.8:
        return pd.to_datetime(s.str.slice(0, 10), format='%Y-%m-%d', errors='coerce')
    if probe.str.match(r'^\d{1,2}\.\d{1,2}\.\d{4}').mean() > 0.8:
        return pd.to_datetime(s.str.slice(0, 10), format='%d.%m.%Y', errors='coerce')
    m = probe.str.extract(r'^(\d{1,2})/(\d{1,2})/(\d{4})')
    if m[0].notna().mean() > 0.8:
        first, second = m[0].astype(float), m[1].astype(float)
        dayfirst = (first > 12).any() and not (second > 12).any()
        part = s.str.split(' ').str[0]
        return pd.to_datetime(part, format='%d/%m/%Y' if dayfirst else '%m/%d/%Y', errors='coerce')
    return pd.to_datetime(s, errors='coerce', dayfirst=True)


def _well(s):
    return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


# ---------- три таблицы ----------
def parse_production(df, oil_density=0.85):
    """-> (добыча, закачка из того же файла или None)."""
    c = find_columns(df, ['well', 'date', 'ql', 'qo', 'wc', 'inj', 'bhp', 'role'])
    for k, title in (('well', 'номер скважины'), ('date', 'дата'), ('ql', 'дебит жидкости')):
        if k not in c:
            raise DataError(f'Добыча: не найден столбец «{title}». Есть столбцы: {", ".join(map(str, df.columns))}')
    out = pd.DataFrame({'well': _well(df[c['well']]), 'date': to_date(df[c['date']]), 'ql': to_number(df[c['ql']])})
    if 'wc' in c:
        wc = to_number(df[c['wc']])
        if wc.max() <= 1.0:
            wc = wc * 100.0
        out['wc'] = wc
    elif 'qo' in c:
        qo = to_number(df[c['qo']])
        if re.search(r'т\s*/|t\s*/|тонн', _norm(c['qo'])):
            qo = qo / oil_density
        out['wc'] = (100.0 * (1.0 - qo / out['ql'].where(out['ql'] > 0))).clip(0, 100)
    else:
        out['wc'] = np.nan
    out['bhp'] = to_number(df[c['bhp']]) if 'bhp' in c else np.nan
    out['inj'] = to_number(df[c['inj']]) if 'inj' in c else np.nan
    if 'role' in c:
        r = df[c['role']].astype(str).str.lower()
        is_inj = r.str.contains('наг|inj|ппд', regex=True)
    else:
        is_inj = (out['inj'] > 0) & ~(out['ql'] > 0)
    out = out.dropna(subset=['date'])
    is_inj = is_inj.reindex(out.index)
    prod = out[~is_inj].assign(role='НЕФ', inj=np.nan)
    inj = out[is_inj & out['inj'].notna()].assign(role='НАГ', ql=np.nan, wc=np.nan, bhp=np.nan)
    return prod, (inj if len(inj) else None)


def parse_injection(df):
    c = find_columns(df, ['well', 'date', 'inj'])
    if 'date' not in c:
        raise DataError(f'Закачка: не найден столбец «дата». Есть столбцы: {", ".join(map(str, df.columns))}')
    if 'well' not in c:                       # «широкая» таблица: дата + столбец на скважину
        wide = df.set_index(c['date'])
        long = wide.stack().reset_index()
        long.columns = ['date', 'well', 'inj']
        out = pd.DataFrame({'well': _well(long['well']), 'date': to_date(long['date']), 'inj': to_number(long['inj'])})
    else:
        if 'inj' not in c:
            raise DataError(f'Закачка: не найден столбец «приёмистость». Есть столбцы: {", ".join(map(str, df.columns))}')
        out = pd.DataFrame({'well': _well(df[c['well']]), 'date': to_date(df[c['date']]),
                            'inj': to_number(df[c['inj']])})
    out = out.dropna(subset=['date', 'inj'])
    return out.assign(role='НАГ', ql=np.nan, wc=np.nan, bhp=np.nan)


def parse_coords(df):
    cols = list(df.columns)
    up = [str(x).strip().upper() for x in cols]

    def pick(letter, words):
        for col, n in zip(cols, up):
            if n == letter or n.endswith(' ' + letter) or n.endswith('_' + letter) or n.startswith(letter + ',') \
                    or any(w in n for w in words):
                return col
        return None

    cx, cy = pick('X', ('EAST', 'ВОСТОК')), pick('Y', ('NORTH', 'СЕВЕР'))
    if cx is None or cy is None:
        raise DataError(f'Координаты: не найдены столбцы X и Y. Есть столбцы: {", ".join(map(str, cols))}')
    cw = find_columns(df, ['well']).get('well') or [c for c in cols if c not in (cx, cy)][0]
    out = pd.DataFrame({'X': to_number(df[cx]).values, 'Y': to_number(df[cy]).values}, index=_well(df[cw]).values)
    out = out.dropna()
    return out[~out.index.duplicated()]


def detect_step(prod):
    """Типичный интервал между записями по скважине, сут -> шаг расчёта (7 или 30)."""
    d = prod.sort_values(['well', 'date']).groupby('well')['date'].diff().dt.days.dropna()
    d = d[d > 0]
    med = float(d.median()) if len(d) else 1.0
    return (30 if med >= 20 else 7), med


def assemble(prod_df, inj_df, coord_df, oil_density=0.85):
    """Три «сырые» таблицы -> (замеры в формате build_dataset, координаты, сведения)."""
    prod, inj_same = parse_production(prod_df, oil_density)
    if inj_df is not None:
        inj = parse_injection(inj_df)
    elif inj_same is not None:
        inj = inj_same
    else:
        raise DataError('Нет данных закачки: задайте файл закачки или файл добычи со столбцами '
                        '«Характер работы» и «Приемистость».')
    coords = parse_coords(coord_df)
    cols = ['well', 'date', 'role', 'ql', 'wc', 'inj', 'bhp']
    df = pd.concat([prod[cols], inj[cols]], ignore_index=True)
    wells_p, wells_i = set(prod['well']), set(inj['well'])
    no_xy = sorted((wells_p | wells_i) - set(coords.index))
    info = {
        'n_prod_rows': len(prod), 'n_inj_rows': len(inj),
        'wells_prod': len(wells_p), 'wells_inj': len(wells_i),
        'no_coords': no_xy, 'has_oil': bool(prod['wc'].notna().any()),
        'date_min': df['date'].min(), 'date_max': df['date'].max(),
    }
    if not len(prod) or not prod['ql'].notna().any():
        raise DataError('Добыча: нет ни одной записи с дебитом жидкости.')
    return df, coords, info
