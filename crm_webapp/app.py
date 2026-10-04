"""Веб-интерфейс к модели CM + ХВ + нейросеть: связи скважин, прогноз, отклик на закачку, сценарии.

Запуск:  python app.py   (или двойной щелчок по run.bat), затем открыть http://127.0.0.1:8050

На входе — данные добычи, закачки и координаты скважин (см. README.md и папку sample_data).
Расчёты — в core/engine.py (пакет hybrid из соседней папки crm_hybrid_model),
графики — в core/figures.py, здесь только разметка страницы и обработчики.
"""
import os
import threading
import traceback
import warnings
import webbrowser

import numpy as np

try:                                              # Dash 2.x / 3.x
    from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
except ImportError:                               # Dash 1.x
    from dash import Dash, no_update, callback_context
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output, State

import config
from core import engine, figures, reader

warnings.filterwarnings('ignore')

STATE = {'model': None, 'run': 0, 'fc': None}     # приложение однопользовательское: одна модель в памяти
SAMPLE = 'Пример: sample_data'
FILES = '__files__'
GRAPH_CFG = {'displaylogo': False, 'scrollZoom': True,
             'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d']}
TAB = {'backgroundColor': '#0d0d0d', 'color': '#c3c2b7', 'padding': '10px 16px', 'borderTop': 'none',
       'borderLeft': 'none', 'borderRight': 'none', 'borderBottom': '2px solid #2c2c2a'}
TAB_ON = dict(TAB, color='#ffffff', backgroundColor='#1a1a19', borderBottom='2px solid #3987e5', fontWeight=600)
TAB_COLORS = {'border': '#2c2c2a', 'primary': '#3987e5', 'background': '#0d0d0d'}

app = Dash(__name__, title='CM: закачка и добыча')
server = app.server


# ============================================================ элементы страницы
def num(v, nd=1, sign=False):
    if v is None or not np.isfinite(v):
        return '—'
    s = f'{v:+,.{nd}f}' if sign else f'{v:,.{nd}f}'
    return s.replace(',', ' ').replace('.', ',').replace('-', '−')


def card(title, *children, sub=None, cls=''):
    head = [html.Div(title, className='card-title')]
    if sub:
        head.append(html.Div(sub, className='card-sub'))
    return html.Div(head + list(children), className=('card ' + cls).strip())


def graph(gid, height):
    return dcc.Graph(id=gid, figure=figures.empty(), config=GRAPH_CFG, style={'height': f'{height}px'})


def table(header, rows, align=None):
    """Простая таблица: header — подписи столбцов, rows — списки ячеек (уже отформатированных)."""
    align = align or ['left'] + ['right'] * (len(header) - 1)
    th = [html.Th(h, style={'textAlign': a}) for h, a in zip(header, align)]
    body = [html.Tr([html.Td(c, style={'textAlign': a}) for c, a in zip(r, align)]) for r in rows]
    return html.Table([html.Thead(html.Tr(th)), html.Tbody(body)], className='tbl')


def tile(label, value, sub=''):
    return html.Div([html.Div(label, className='tile-label'), html.Div(value, className='tile-value'),
                     html.Div(sub, className='tile-sub')], className='tile')


def field(label, control, hint=None):
    kids = [html.Label(label, className='lbl'), control]
    if hint:
        kids.append(html.Div(hint, className='hint'))
    return html.Div(kids, className='field')


def upload(uid, text):
    return html.Div([
        dcc.Upload(id=uid, children=html.Div([html.Span(text, className='up-title'),
                                              html.Span('выберите файл', className='up-hint')]),
                   className='upload', multiple=False,
                   style_active={'borderStyle': 'dashed', 'borderColor': '#3987e5', 'backgroundColor': '#262624'}),
        html.Div(id=uid + '-name', className='up-name'),
    ])


def source_options():
    opts = [{'label': 'Свои файлы', 'value': FILES}]
    if os.path.isfile(os.path.join(config.HERE, 'sample_data', 'production.csv')):
        opts.append({'label': SAMPLE, 'value': SAMPLE})
    try:
        opts += [{'label': k, 'value': k} for k in engine.demo_blocks()]
    except Exception:
        pass
    return opts


HORIZONS = [{'label': '3 месяца', 'value': 91}, {'label': '6 месяцев', 'value': 182},
            {'label': '12 месяцев', 'value': 364}]


def side_panel():
    opts = source_options()
    return html.Div([
        html.Div('Данные', className='side-title'),
        field('Источник', dcc.Dropdown(id='src', options=opts, value=FILES, clearable=False),
              'В списке есть готовые примеры — можно начать с них.' if len(opts) > 1 else None),
        html.Div([
            upload('up-prod', 'Добыча'),
            upload('up-inj', 'Закачка'),
            upload('up-xy', 'Координаты'),
        ], id='uploads'),
        field('Шаг расчёта', dcc.RadioItems(
            id='step', value='auto', className='radio',
            options=[{'label': 'определить по данным', 'value': 'auto'},
                     {'label': '7 сут (суточные замеры)', 'value': '7'},
                     {'label': '30 сут (месячные данные)', 'value': '30'}])),
        dcc.Checklist(id='calib', value=[], className='radio',
                      options=[{'label': 'калибровать отклик по этим данным (дольше)', 'value': 'on'}]),
        html.Button('Загрузить и рассчитать', id='btn-load', className='btn primary'),
        dcc.Loading(html.Div('Данные не загружены.', id='status', className='status'), color='#3987e5'),

        html.Div('Скважины', className='side-title'),
        field('Выбор', dcc.Dropdown(id='wells', options=[], value=[], multi=True, placeholder='все скважины'),
              'Можно щёлкнуть по скважине на карте; убрать — крестиком в списке. '
              'Выбор фильтрует связи и график динамики.'),

        html.Div('Связи на карте', className='side-title'),
        field('Коэффициент', dcc.RadioItems(
            id='coef-src', value='g', className='radio',
            options=[{'label': 'g — отклик, калиброванный по изменениям закачки', 'value': 'g'},
                     {'label': 'f — доли закачки по CM', 'value': 'f'}])),
        field('Порог, доля от наибольшего коэффициента',
              dcc.Slider(id='min-coef', min=0, max=1, step=0.05, value=0.15,
                         marks={0: '0', 0.25: '0,25', 0.5: '0,5', 0.75: '0,75', 1: '1'})),
        field('Расстояние между скважинами не больше, м',
              dcc.Slider(id='max-dist', min=250, max=3000, step=250, value=1000,
                         marks={250: '250', 1000: '1000', 2000: '2000', 3000: '3000'})),
    ], className='side')


def tab_map():
    return html.Div([
        html.Div([
            card('Карта связей «нагнетательная → добывающая»', dcc.Graph(
                id='map', figure=figures.empty(), config=GRAPH_CFG, style={'height': '540px'}),
                sub='Толщина и яркость линии — величина коэффициента. Колесо мыши — масштаб.'),
            card('Самые сильные связи', html.Div(id='pairs', className='scroll', style={'maxHeight': '540px'})),
        ], className='grid-2-1'),
        card('Динамика', graph('hist', 280),
             sub='Сумма по выбранным скважинам; если ничего не выбрано — по участку в целом.'),
    ], className='tab-body')


def tab_forecast():
    return html.Div([
        html.Div([
            field('Дата прогноза', dcc.Dropdown(id='fc-date', options=[], clearable=False)),
            field('Горизонт', dcc.Dropdown(id='fc-h', options=HORIZONS, value=182, clearable=False)),
            dcc.Checklist(id='fc-opts', value=['oil'], className='radio', options=[
                {'label': 'прогноз нефти по характеристикам вытеснения (≈ 1 с на скважину)', 'value': 'oil'},
                {'label': 'нейросетевая поправка (от десятков секунд до нескольких минут)', 'value': 'nn'}]),
            html.Button('Рассчитать прогноз', id='btn-fc', className='btn primary'),
        ], className='controls'),
        dcc.Loading(html.Div(className='stack', children=[
            html.Div(id='fc-note', className='note'),
            field('Показать', dcc.Dropdown(id='fc-well', options=[{'label': 'Базовый фонд — сумма', 'value': '*'}],
                                           value='*', clearable=False), ),
            html.Div([
                card('Жидкость', graph('fc-liq', 340)),
                card('Нефть', graph('fc-oil', 340)),
            ], className='grid-1-1'),
            card('Точность на проверочном периоде', html.Div(id='fc-metrics', className='scroll'),
                 sub='Считается там, где есть факт. wMAPE — сумма модулей ошибок, отнесённая к сумме факта.'),
        ]), color='#3987e5'),
    ], className='tab-body')


def tab_response():
    return html.Div([dcc.Loading(html.Div([
        html.Div(id='resp-note', className='note'),
        html.Div([
            card('Отклик соседней добывающей на изменение закачки', graph('resp-fig', 340),
                 html.Div(id='resp-summary', className='scroll'),
                 sub='В расчёте на одну соседнюю скважину (ближе 1000 м), работавшую без остановок, '
                     'за вычетом изменения у дальних скважин участка.'),
            card('Нагнетательные скважины', html.Div(id='resp-inj', className='scroll', style={'maxHeight': '560px'}),
                 sub='Σg — сколько м³/сут жидкости даёт соседям 1 м³/сут закачки по модели отклика; '
                     'Σf — то же по CM без калибровки; «по нефти» — Σg с учётом текущей доли нефти соседей.'),
        ], className='grid-1-1'),
    ], className='stack'), color='#3987e5')], className='tab-body')


def tab_scenario():
    return html.Div([
        html.Div([
            field('Способ', dcc.RadioItems(
                id='scn-mode', value='manual', className='radio',
                options=[{'label': 'задать вручную', 'value': 'manual'},
                         {'label': 'подобрать перераспределение', 'value': 'auto'}])),
            html.Div([
                field('Уменьшить закачку на скважинах', dcc.Dropdown(id='don', options=[], value=[], multi=True)),
                field('Увеличить закачку на скважинах', dcc.Dropdown(id='rec', options=[], value=[], multi=True)),
                field('Объём, м³/сут', dcc.Input(id='vol', type='number', value=100, min=0, className='inp'),
                      'Снимается пропорционально текущей приёмистости, раздаётся поровну. '
                      'Если заполнен только один список — закачка просто уменьшается или растёт.'),
            ], id='scn-manual', className='controls-inner'),
            html.Div([
                field('Предел изменения по скважине, %', dcc.Input(id='limit', type='number', value=20, min=1, max=100,
                                                                    className='inp')),
                field('Цель', dcc.RadioItems(id='target', value='oil', className='radio', options=[
                    {'label': 'больше нефти', 'value': 'oil'}, {'label': 'больше жидкости', 'value': 'liq'}]),
                      'Суммарная закачка не меняется.'),
            ], id='scn-auto', className='controls-inner', style={'display': 'none'}),
            field('Горизонт', dcc.Dropdown(id='scn-h', options=HORIZONS, value=182, clearable=False)),
            html.Button('Рассчитать сценарий', id='btn-scn', className='btn primary'),
        ], className='controls'),
        dcc.Loading(html.Div(className='stack', children=[
            html.Div(id='scn-note', className='note'),
            html.Div(id='scn-tiles', className='tiles'),
            html.Div([
                card('Изменение добычи участка во времени', graph('scn-time', 320)),
                card('Добывающие с наибольшим изменением', graph('scn-wells', 320)),
            ], className='grid-1-1'),
            card('Изменение закачки по скважинам', html.Div(id='scn-inj', className='scroll',
                                                           style={'maxHeight': '320px'})),
        ]), color='#3987e5'),
    ], className='tab-body')


app.layout = html.Div([
    dcc.Store(id='run', data=0),
    dcc.Store(id='fc-run', data=0),
    html.Div([
        html.Div('Закачка и добыча: связи, прогноз, перераспределение', className='title'),
        html.Div('Модель CM + характеристики вытеснения + нейросетевая поправка', id='header-sub',
                 className='subtitle'),
    ], className='header'),
    side_panel(),
    html.Div(dcc.Tabs(id='tabs', value='map', colors=TAB_COLORS, children=[
        dcc.Tab(label='Карта связей', value='map', children=tab_map(), style=TAB, selected_style=TAB_ON),
        dcc.Tab(label='Прогноз', value='fc', children=tab_forecast(), style=TAB, selected_style=TAB_ON),
        dcc.Tab(label='Отклик на закачку', value='resp', children=tab_response(), style=TAB,
                selected_style=TAB_ON),
        dcc.Tab(label='Сценарий', value='scn', children=tab_scenario(), style=TAB, selected_style=TAB_ON),
    ]), className='main'),
], className='app')


# ============================================================ обработчики
def model():
    return STATE['model']


def triggered():
    t = callback_context.triggered
    return t[0]['prop_id'].split('.')[0] if t else ''


def error_box(text):
    return html.Div(text, className='error')


for _uid in ('up-prod', 'up-inj', 'up-xy'):
    @app.callback(Output(_uid + '-name', 'children'), [Input(_uid, 'filename')])
    def _show_name(name):
        return name or ''


@app.callback(Output('uploads', 'style'), [Input('src', 'value')])
def toggle_uploads(src):
    return {} if src == FILES else {'display': 'none'}


@app.callback(
    [Output('run', 'data'), Output('status', 'children')],
    [Input('btn-load', 'n_clicks')],
    [State('src', 'value'), State('up-prod', 'contents'), State('up-prod', 'filename'),
     State('up-inj', 'contents'), State('up-inj', 'filename'),
     State('up-xy', 'contents'), State('up-xy', 'filename'), State('step', 'value'), State('calib', 'value')])
def load_data(n, src, pc, pn, ic, iname, xc, xn, step, calib):
    if not n:
        return no_update, no_update
    step = None if step == 'auto' else int(step)
    cal = 'on' in (calib or [])
    try:
        if src == FILES:
            if not pc or not xc:
                return no_update, error_box('Нужны как минимум файл добычи и файл координат.')
            prod = reader.read_table(reader.decode_upload(pc), pn)
            inj = reader.read_table(reader.decode_upload(ic), iname) if ic else None
            xy = reader.read_table(reader.decode_upload(xc), xn)
            m = engine.load_files(prod, inj, xy, os.path.splitext(pn or 'данные')[0], step, cal)
        elif src == SAMPLE:
            d = os.path.join(config.HERE, 'sample_data')
            m = engine.load_files(reader.read_path(os.path.join(d, 'production.csv')),
                                  reader.read_path(os.path.join(d, 'injection.csv')),
                                  reader.read_path(os.path.join(d, 'coordinates.csv')), 'sample_data', step, cal)
        else:
            m = engine.load_demo(src, step, cal)
    except reader.DataError as e:
        return no_update, error_box(str(e))
    except Exception as e:                                   # показать причину, а не «ничего не произошло»
        traceback.print_exc()
        return no_update, error_box(f'Ошибка при загрузке: {type(e).__name__}: {e}')
    STATE.update(model=m, fc=None, run=STATE['run'] + 1)
    i = m.info
    lines = [
        f'{m.name}: добывающих {i["n_prod"]}, нагнетательных {i["n_inj"]}.',
        f'История {i["date_min"]:%d.%m.%Y} – {i["date_max"]:%d.%m.%Y}, {i["T"]} шагов по {i["step"]} сут.',
        f'CM настроена: wMAPE по жидкости {100 * i["cm_wmape"]:.1f} %'.replace('.', ',') + '.',
        ('Отклик: откалиброван ' + i['calib_note'] if m.calibrated else
         'Отклик: коэффициенты по умолчанию (5 участков, 647 изменений закачки)') +
        f': τ = {m.rm.tau:.0f} сут, G_geo = {m.rm.g_geo:.3f}, G_f = {m.rm.g_f:.3f}.'.replace('.', ',', 2),
    ]
    if i.get('calib_note') and not m.calibrated:
        lines.append(i['calib_note'].capitalize() + '.')
    if i.get('no_coords'):
        lines.append(f'Без координат (не вошли в расчёт): {len(i["no_coords"])} скв.')
    skipped = i.get('wells_prod', 0) + i.get('wells_inj', 0) - i['n_prod'] - i['n_inj']
    if skipped > 0:
        lines.append(f'Не вошли в расчёт (нет координат или короткая история): {skipped} скв.')
    if not i['has_oil']:
        lines.append('Обводнённость не задана — расчёты по нефти недоступны.')
    return STATE['run'], [html.Div(x) for x in lines]


@app.callback([Output('wells', 'options'), Output('wells', 'value')],
              [Input('run', 'data'), Input('map', 'clickData')], [State('wells', 'value')])
def select_wells(run, click, value):
    m = model()
    if m is None:
        return [], []
    opts = ([{'label': f'доб. {w}', 'value': w} for w in m.ds.prods] +
            [{'label': f'нагн. {w}', 'value': w} for w in m.ds.injs if w not in m.ds.prods])
    if triggered() != 'map' or not click:
        return opts, []
    w = click['points'][0].get('customdata')
    value = list(value or [])
    if w is None:
        return opts, value
    value = [v for v in value if v != w] if w in value else value + [w]
    return opts, value


@app.callback([Output('map', 'figure'), Output('pairs', 'children')],
              [Input('run', 'data'), Input('wells', 'value'), Input('coef-src', 'value'),
               Input('min-coef', 'value'), Input('max-dist', 'value')])
def update_map(run, wells, src, share, max_dist):
    m = model()
    if m is None:
        return figures.empty(), 'Нет данных.'
    wells = [w for w in (wells or []) if w in m.ds.prods or w in m.ds.injs]
    full = engine.pairs_table(m, max_dist=max_dist, source=src, wells=wells)
    cmax = float(full['coef'].max()) if len(full) else 0.0
    pairs = full[full['coef'] >= (share or 0) * cmax]
    rows = [[r.inj, r.prod, num(r.dist, 0), num(r.g, 3), num(r.f, 3)] for r in pairs.head(60).itertuples()]
    tbl = [table(['Нагн.', 'Доб.', 'м', 'g', 'f (CM)'], rows, ['left', 'left', 'right', 'right', 'right']),
           html.Div(f'Показано {len(rows)} из {len(pairs)} связей, по убыванию '
                    f'{"g" if src == "g" else "f"}.', className='hint')]
    return figures.well_map(m, pairs, wells, src), tbl


@app.callback(Output('hist', 'figure'), [Input('run', 'data'), Input('wells', 'value')])
def update_history(run, wells):
    m = model()
    return figures.empty() if m is None else figures.history(m, wells)


@app.callback([Output('fc-date', 'options'), Output('fc-date', 'value'),
               Output('don', 'options'), Output('rec', 'options'), Output('don', 'value'), Output('rec', 'value'),
               Output('header-sub', 'children')], [Input('run', 'data')])
def fill_controls(run):
    m = model()
    if m is None:
        return [], None, [], [], [], [], no_update
    ds = m.ds
    every = 4 if ds.step < 20 else 1
    first = min(52 if ds.step < 20 else 24, ds.T - 1)
    dates = [{'label': f'{ds.dates[t]:%d.%m.%Y} (проверка на истории)', 'value': t}
             for t in range(ds.T - every, first, -every)]
    dates = [{'label': 'конец истории — прогноз вперёд', 'value': ds.T}] + dates
    back = [d['value'] for d in dates if d['value'] <= ds.T - int(round(182 / ds.step))]
    default = max(back) if back else dates[-1]['value']        # полгода до конца истории: весь прогноз сверяется с фактом
    now = m.inj_now
    inj = [{'label': f'{w} ({num(q, 0)} м³/сут)', 'value': w} for w, q in zip(ds.injs, now)]
    sub = (f'{m.name}: {len(ds.prods)} добывающих, {len(ds.injs)} нагнетательных, '
           f'{ds.dates[0]:%m.%Y} – {m.info["date_max"]:%m.%Y}')
    return dates, default, inj, inj, [], [], sub


@app.callback([Output('fc-run', 'data'), Output('fc-note', 'children'), Output('fc-metrics', 'children'),
               Output('fc-well', 'options'), Output('fc-well', 'value')],
              [Input('btn-fc', 'n_clicks'), Input('run', 'data')],
              [State('fc-date', 'value'), State('fc-h', 'value'), State('fc-opts', 'value'), State('fc-run', 'data')])
def run_forecast(n, run, t0, days, opts, fc_run):
    m = model()
    all_opt = [{'label': 'Базовый фонд — сумма', 'value': '*'}]
    if m is None or triggered() != 'btn-fc' or t0 is None:
        STATE['fc'] = None
        text = 'Выберите дату и нажмите «Рассчитать прогноз».' if m is not None else 'Сначала загрузите данные.'
        return (fc_run or 0) + 1, text, '', all_opt, '*'
    ds = m.ds
    h = max(int(round(days / ds.step)), 2)
    try:
        fc = engine.forecast(m, int(t0), h, use_nn='nn' in (opts or []), with_oil='oil' in (opts or []))
    except Exception as e:
        traceback.print_exc()
        STATE['fc'] = None
        return (fc_run or 0) + 1, error_box(f'Ошибка расчёта: {type(e).__name__}: {e}'), '', all_opt, '*'
    STATE['fc'] = fc
    n_fund, late = int(fc['fund'].sum()), int((~fc['fund'] & (fc['up_f'].sum(axis=0) > 0)).sum())
    note = [f'Прогноз от {fc["dates"][0]:%d.%m.%Y} на {h} шагов по {ds.step} сут. '
            f'Базовый фонд — {n_fund} скв., работавших на дату прогноза.']
    if late:
        note.append(f' Запущены позже и в сумму не входят: {late} скв.')
    if fc['n_fact'] < h:
        note.append(' За концом истории закачка и график работы скважин оставлены на последнем уровне.')
    else:
        note.append(' Закачка и график работы на прогнозном периоде — фактические, дебиты моделям неизвестны.')
    note += [' ' + x for x in fc['notes']]
    opts_w = all_opt + [{'label': f'скв. {w}', 'value': w} for w, ok in zip(ds.prods, fc['fund']) if ok]
    return (fc_run or 0) + 1, ''.join(note), metrics_table(fc), opts_w, '*'


def metrics_table(fc):
    mt = fc['metrics']
    if mt is None:
        return html.Div('Проверочного периода нет: прогноз начинается с конца истории.', className='hint')
    rows = [[r.kind, figures.METHOD[r.method][0], num(100 * r.block_wmape), num(100 * r.block_cum, 1, True),
             num(100 * r.well_wmape), str(r.n_wells)] for r in mt.itertuples()]
    return table(['Показатель', 'Вариант', 'wMAPE суммы по фонду, %', 'Ошибка накопленной, %',
                  'Медиана wMAPE по скважинам, %', 'Скважин'], rows,
                 ['left', 'left', 'right', 'right', 'right', 'right'])


@app.callback([Output('fc-liq', 'figure'), Output('fc-oil', 'figure')],
              [Input('fc-run', 'data'), Input('fc-well', 'value')])
def show_forecast(fc_run, well):
    m, fc = model(), STATE['fc']
    if m is None or fc is None:
        e = figures.empty('Прогноз ещё не рассчитан')
        return e, e
    return figures.forecast(m, fc, well, 'liq'), figures.forecast(m, fc, well, 'oil')


@app.callback([Output('resp-fig', 'figure'), Output('resp-summary', 'children'), Output('resp-inj', 'children'),
               Output('resp-note', 'children')], [Input('tabs', 'value'), Input('run', 'data')])
def update_response(tab, run):
    m = model()
    if m is None:
        return figures.empty(), '', '', 'Сначала загрузите данные.'
    if tab != 'resp':
        return no_update, no_update, no_update, no_update
    try:
        es = engine.event_study(m)
    except Exception as e:
        traceback.print_exc()
        return figures.empty('Ошибка расчёта'), '', '', error_box(f'{type(e).__name__}: {e}')
    s = es['summary']
    note = (f'Найдено резких изменений приёмистости (остановы, пуски, ступени): {es["n_events"]}. '
            'На одном участке событий мало, поэтому интервалы широкие. Для сравнения, по 5 участкам '
            '(647 событий) отклик жидкости составил +0,60 / +1,01 / +1,46 м³/сут через 1–8 / 9–16 / 17–26 недель, '
            'отклик по нефти статистически не подтверждён, а CM без калибровки завышала отклик в 2–3 раза. '
            'CM и модель отклика здесь настроены на всей истории участка, то есть сравнение не является '
            'проверкой на новых данных.')
    if s is None:
        summ = html.Div('Недостаточно событий для оценки.', className='hint')
    else:
        rows = [[r.window, f'{num(r.fact, 2, True)} [{num(r.lo, 2, True)}; {num(r.hi, 2, True)}]',
                 f'{num(r.oil, 2, True)} [{num(r.oil_lo, 2, True)}; {num(r.oil_hi, 2, True)}]',
                 num(r.rm, 2, True), num(r.cm, 2, True), str(r.n_ev), str(r.n_pairs)] for r in s.itertuples()]
        summ = [table(['Окно', 'Факт, жидкость', 'Факт, нефть', 'Модель', 'CM', 'Событий', 'Пар'], rows),
                html.Div('м³/сут на +100 м³/сут закачки; в скобках 95 %-й интервал (бутстреп по событиям).',
                         className='hint')]
    it = engine.injector_table(m).sort_values('eff_oil', ascending=False)
    n_ev = es['events'].groupby('injector').size() if es['n_events'] else {}
    rows = [[r.inj, num(r.rate, 0), str(r.n_near), num(r.sum_g, 3), num(r.sum_f, 3), num(r.eff_oil, 3),
             str(int(n_ev.get(r.inj, 0)))] for r in it.itertuples()]
    inj = table(['Нагн.', 'Закачка, м³/сут', 'Соседей ≤ 1000 м', 'Σg', 'Σf (CM)', 'Σg по нефти', 'Событий'], rows)
    return figures.response(s), summ, inj, note


@app.callback([Output('scn-manual', 'style'), Output('scn-auto', 'style')], [Input('scn-mode', 'value')])
def toggle_mode(mode):
    show, hide = {}, {'display': 'none'}
    return (show, hide) if mode == 'manual' else (hide, show)


@app.callback([Output('scn-tiles', 'children'), Output('scn-time', 'figure'), Output('scn-wells', 'figure'),
               Output('scn-inj', 'children'), Output('scn-note', 'children')],
              [Input('btn-scn', 'n_clicks'), Input('run', 'data')],
              [State('scn-mode', 'value'), State('don', 'value'), State('rec', 'value'), State('vol', 'value'),
               State('limit', 'value'), State('target', 'value'), State('scn-h', 'value')])
def run_scenario(n, run, mode, don, rec, vol, limit, target, days):
    m = model()
    e = figures.empty('Сценарий ещё не рассчитан')
    if m is None or triggered() != 'btn-scn':
        return [], e, e, '', ('Задайте изменение закачки и нажмите «Рассчитать сценарий».' if m is not None
                              else 'Сначала загрузите данные.')
    ds = m.ds
    if mode == 'manual':
        if not don and not rec:
            return [], e, e, '', error_box('Выберите хотя бы одну нагнетательную скважину.')
        delta = engine.manual_delta(m, don, rec, vol or 0)
    else:
        delta = engine.suggest_delta(m, (limit or 20) / 100.0, target)
    h = max(int(round(days / ds.step)), 2)
    s = engine.scenario(m, delta, h)
    if np.abs(s['delta']).sum() < 1e-6:
        return [], e, e, '', error_box('Изменение закачки нулевое: выбранные скважины сейчас не работают '
                                       'или объём равен нулю.')
    months = round(h * ds.step / 30.4)
    tiles = [
        tile('Перераспределено закачки', num(s['moved'], 0) + ' м³/сут',
             f'суммарная закачка: {num(s["inj_change"], 0, True)} м³/сут'),
        tile(f'Жидкость через {months} мес.', num(s['liq_end'], 1, True) + ' м³/сут', 'по модели отклика'),
        tile(f'Нефть через {months} мес.', num(s['oil_end'], 1, True) + ' м³/сут',
             f'накопленная за период: {num(s["oil_cum"], 0, True)} м³'),
        tile('Жидкость по CM без калибровки', num(s['cm_end'], 1, True) + ' м³/сут', 'для сравнения; завышает отклик'),
    ]
    now = m.inj_now
    order = np.argsort(s['delta'])
    rows = [[ds.injs[i], num(now[i], 0), num(now[i] + s['delta'][i], 0), num(s['delta'][i], 0, True),
             num(float((m.g[i] * m.up_now * m.oil_frac).sum()), 3)]
            for i in order if abs(s['delta'][i]) >= 0.5]
    tbl = table(['Нагн.', 'Было, м³/сут', 'Стало, м³/сут', 'Изменение', 'Σg по нефти'], rows)
    note = ('Эффект считается по модели отклика, калиброванной по фактическим изменениям закачки; запаздывание '
            f'τ = {m.rm.tau:.0f} сут. Изменение нефти = изменение жидкости × текущая доля нефти скважины: '
            'прямой отклик по нефти на проверке статистически не подтверждён, поэтому это оценка порядка величины, '
            'а не гарантированный прирост. Суммарное изменение жидкости складывается из приростов и потерь по '
            'скважинам, поэтому нефть может измениться сильнее жидкости. Ограничения по давлению, приёмистости и риску прорыва воды не учитываются.')
    return tiles, figures.scenario_time(s), figures.scenario_wells(m, s), tbl, note


if __name__ == '__main__':
    url = f'http://127.0.0.1:{config.PORT}'
    print(f'\nИнтерфейс: {url}   (остановить — Ctrl+C)\n')
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    run = getattr(app, 'run', None) or app.run_server          # Dash 1.x: run_server
    run(debug=False, port=config.PORT)
