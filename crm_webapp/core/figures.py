"""Графики в виде словарей Plotly (без импорта plotly — рисует их Dash в браузере).

Цвет закреплён за сущностью и одинаков на всех графиках:
    закачка и нагнетательные — синий, нефть и добывающие — оранжевый, жидкость — зелёный;
    факт — светло-серый; инерционный прогноз — жёлтый, CM — пурпурный, итоговая модель — фиолетовый.
"""
import numpy as np

SURFACE, PAGE = '#1a1a19', '#0d0d0d'
INK, INK2, MUTED, GRID, AXIS = '#ffffff', '#c3c2b7', '#898781', '#2c2c2a', '#383835'
BLUE, ORANGE, AQUA = '#3987e5', '#d95926', '#199e70'
YELLOW, MAGENTA, VIOLET = '#c98500', '#d55181', '#9085e9'
EDGE_RAMP = ['#184f95', '#256abf', '#5598e7', '#9ec5f4']       # слабая -> сильная связь
EDGE_WIDTH = [1.0, 1.8, 3.0, 4.5]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

METHOD = {
    'naive':  ('Инерционный (уровень на дату прогноза)', YELLOW, 'dot'),
    'cm':     ('CM: уровень + изменение по модели', MAGENTA, 'dash'),
    'hybrid': ('Гибрид: CM + нейросетевая поправка', VIOLET, 'solid'),
    'dc':     ('ХВ × прогноз жидкости', VIOLET, 'solid'),
}


def _f(a, nd=2):
    """numpy -> список обычных чисел (None вместо NaN), чтобы словарь сериализовался в JSON."""
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in np.asarray(a, float).ravel()]


def _d(dates):
    return [str(np.datetime_as_string(np.datetime64(x), unit='D')) for x in dates]


def layout(y_title=None, x_title=None, legend=True, **kw):
    lay = dict(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, separators=', ',
        font=dict(family=FONT, color=INK2, size=12),
        margin=dict(l=58, r=14, t=34 if legend else 12, b=44 if x_title else 34),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                   title=dict(text=x_title, font=dict(color=MUTED, size=12)) if x_title else None),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), rangemode='tozero',
                   title=dict(text=y_title, font=dict(color=MUTED, size=12)) if y_title else None),
        showlegend=legend,
        legend=dict(orientation='h', x=0, y=1.02, yanchor='bottom', font=dict(color=INK2, size=12),
                    bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#262624', bordercolor=AXIS, font=dict(color=INK, family=FONT, size=12)),
    )
    lay.update(kw)
    return lay


def empty(text='Загрузите данные и нажмите «Загрузить и рассчитать»'):
    lay = layout(legend=False)
    lay['xaxis'] = dict(visible=False)
    lay['yaxis'] = dict(visible=False)
    lay['annotations'] = [dict(text=text, xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                               font=dict(color=MUTED, size=14))]
    return dict(data=[], layout=lay)


def _line(x, y, name, color, dash='solid', width=2, unit='м³/сут'):
    return dict(type='scatter', mode='lines', x=x, y=_f(y, 1), name=name,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate='%{y:,.1f} ' + unit + '<extra>' + name + '</extra>')


# ------------------------------------------------------------ карта связей
def well_map(m, pairs, selected=(), source='g'):
    ds, xy = m.ds, m.coords
    x0, y0 = float(xy['X'].min()), float(xy['Y'].min())
    xp, yp = xy.loc[ds.prods, 'X'].values - x0, xy.loc[ds.prods, 'Y'].values - y0
    xi, yi = xy.loc[ds.injs, 'X'].values - x0, xy.loc[ds.injs, 'Y'].values - y0
    sel = set(selected or ())
    data = []
    sym = 'g' if source == 'g' else 'f'
    if len(pairs):
        cmax = float(pairs['coef'].max())
        cls = np.minimum((pairs['coef'].values / max(cmax, 1e-12) * 4).astype(int), 3)
        for k in range(4):
            part = pairs[cls == k]
            if not len(part):
                continue
            ex, ey = [], []
            for i, j in zip(part['i'], part['j']):
                ex += [float(xi[i]), float(xp[j]), None]
                ey += [float(yi[i]), float(yp[j]), None]
            lo, hi = cmax * k / 4, cmax * (k + 1) / 4
            data.append(dict(type='scatter', mode='lines', x=ex, y=ey, hoverinfo='skip',
                             line=dict(color=EDGE_RAMP[k], width=EDGE_WIDTH[k]),
                             name=f'{sym} {lo:.2f}–{hi:.2f}'.replace('.', ','), legendgroup='edges'))
        data.append(dict(
            type='scatter', mode='markers', showlegend=False,
            x=_f((xi[pairs['i']] + xp[pairs['j']]) / 2, 1), y=_f((yi[pairs['i']] + yp[pairs['j']]) / 2, 1),
            marker=dict(size=12, color='rgba(0,0,0,0)'),
            text=[f'{a} → {b}<br>{sym} = {c:.3f}<br>расстояние {d:.0f} м'
                  for a, b, c, d in zip(pairs['inj'], pairs['prod'], pairs['coef'], pairs['dist'])],
            hovertemplate='%{text}<extra></extra>'))
    liq, of, inj = m.liq_now * m.up_now, m.oil_frac, m.inj_now
    labels = len(ds.prods) + len(ds.injs) <= 70
    k = 1.0 if labels else 0.65                       # на больших участках значки мельче
    mode = 'markers+text' if labels else 'markers'

    def ring(names):
        return dict(color=[INK if w in sel else SURFACE for w in names], width=[3 if w in sel else 2 for w in names])

    data.append(dict(
        type='scatter', mode=mode, name='Добывающие (размер — дебит жидкости)', x=_f(xp, 1), y=_f(yp, 1),
        customdata=list(ds.prods), text=list(ds.prods), textposition='top center',
        textfont=dict(color=MUTED, size=10),
        marker=dict(symbol='circle', color=ORANGE, size=_f(k * (9 + 13 * np.sqrt(liq / max(liq.max(), 1e-9))), 1),
                    opacity=[1.0 if u > 0 else 0.35 for u in m.up_now], line=ring(ds.prods)),
        hovertext=[f'Добывающая {w}<br>жидкость {q:.1f} м³/сут<br>нефть {q * f:.1f} м³/сут<br>'
                   f'обводнённость {100 * (1 - f):.0f} %' + ('' if u > 0 else '<br>не работает')
                   for w, q, f, u in zip(ds.prods, m.liq_now, of, m.up_now)],
        hovertemplate='%{hovertext}<extra></extra>'))
    sg = (m.g * m.up_now).sum(axis=1)
    data.append(dict(
        type='scatter', mode=mode, name='Нагнетательные (размер — приёмистость)', x=_f(xi, 1), y=_f(yi, 1),
        customdata=list(ds.injs), text=list(ds.injs), textposition='bottom center',
        textfont=dict(color=MUTED, size=10),
        marker=dict(symbol='triangle-up', color=BLUE, size=_f(k * (10 + 13 * np.sqrt(inj / max(inj.max(), 1e-9))), 1),
                    opacity=[1.0 if q > 1 else 0.35 for q in inj], line=ring(ds.injs)),
        hovertext=[f'Нагнетательная {w}<br>закачка {q:.1f} м³/сут<br>отклик соседей на 1 м³/сут: {s:.3f}'
                   + ('' if q > 1 else '<br>не работает')
                   for w, q, s in zip(ds.injs, inj, sg)],
        hovertemplate='%{hovertext}<extra></extra>'))
    span = max(float(max(xp.max(), xi.max())), 1.0)
    bar = float(10 ** np.floor(np.log10(span / 4)))
    bar = bar * (5 if span / 4 >= 5 * bar else 2 if span / 4 >= 2 * bar else 1)
    lay = layout(hovermode='closest', dragmode='pan', margin=dict(l=8, r=8, t=34, b=8))
    lay['xaxis'] = dict(visible=False)
    lay['yaxis'] = dict(visible=False, scaleanchor='x', scaleratio=1)
    lay['shapes'] = [dict(type='line', xref='x', yref='paper', x0=0, x1=bar, y0=0.02, y1=0.02,
                          line=dict(color=MUTED, width=2))]
    lay['annotations'] = [dict(xref='x', yref='paper', x=bar / 2, y=0.03, yanchor='bottom', showarrow=False,
                               text=f'{bar:.0f} м', font=dict(color=MUTED, size=11))]
    return dict(data=data, layout=lay)


def history(m, selected=()):
    """Динамика: по выбранным скважинам (сумма) или по участку в целом."""
    ds = m.ds
    x = _d(ds.dates)
    pj = [j for j, w in enumerate(ds.prods) if w in (selected or ())]
    ij = [i for i, w in enumerate(ds.injs) if w in (selected or ())]
    if not pj and not ij:
        pj, ij = list(range(len(ds.prods))), list(range(len(ds.injs)))
    data = []
    if pj:
        data.append(_line(x, ds.liq[:, pj].sum(axis=1), 'Жидкость', AQUA))
        if m.info.get('has_oil'):
            data.append(_line(x, ds.oil[:, pj].sum(axis=1), 'Нефть', ORANGE))
    if ij:
        data.append(_line(x, ds.inj[:, ij].sum(axis=1), 'Закачка', BLUE))
    return dict(data=data, layout=layout(y_title='м³/сут'))


# ------------------------------------------------------------ прогноз
def forecast(m, fc, well=None, kind='liq'):
    ds = m.ds
    t0, n_fact = fc['t0'], fc['n_fact']
    cols = [ds.prods.index(well)] if well in ds.prods else list(np.where(fc['fund'])[0])
    fact = ds.liq if kind == 'liq' else ds.oil
    a = max(t0 - 2 * fc['horizon'], 0)
    b = t0 + n_fact
    data = [_line(_d(ds.dates[a:b]), fact[a:b][:, cols].sum(axis=1), 'Факт', INK2)]
    preds = fc['pred'] if kind == 'liq' else fc['oil']
    x = _d(fc['dates'])
    for k, p in preds.items():
        name, color, dash = METHOD[k]
        data.append(_line(x, p[:, cols].sum(axis=1), name, color, dash))
    lay = layout(y_title='жидкость, м³/сут' if kind == 'liq' else 'нефть, м³/сут')
    lay['shapes'] = [dict(type='line', xref='x', yref='paper', x0=x[0], x1=x[0], y0=0, y1=1,
                          line=dict(color=MUTED, width=1))]
    lay['annotations'] = [dict(xref='x', yref='paper', x=x[0], y=1, xanchor='left', yanchor='top', showarrow=False,
                               text=' дата прогноза', font=dict(color=MUTED, size=11))]
    if not preds:
        return empty('Нет данных по нефти' if kind == 'oil' else 'Нет прогноза')
    return dict(data=data, layout=lay)


# ------------------------------------------------------------ отклик на закачку
def response(summary):
    if summary is None or not len(summary):
        return empty('На участке не найдено достаточно резких изменений закачки')
    x = list(summary['window'])

    def bar(col, name, color, err=False):
        t = dict(type='bar', x=x, y=_f(summary[col]), name=name,
                 marker=dict(color=color, cornerradius=3, line=dict(color=SURFACE, width=1)),
                 hovertemplate='%{y:+.2f} м³/сут<extra>' + name + '</extra>')
        if err:
            t['error_y'] = dict(type='data', symmetric=False, array=_f(summary['hi'] - summary[col]),
                                arrayminus=_f(summary[col] - summary['lo']), color=INK, thickness=1.5, width=5)
        return t

    data = [bar('fact', 'Факт (95 %-й интервал)', INK2, True),
            bar('rm', 'Модель отклика', VIOLET),
            bar('cm', 'CM без калибровки', MAGENTA)]
    lay = layout(y_title='м³/сут жидкости на +100 м³/сут закачки', x_title='время после изменения закачки',
                 barmode='group', bargap=0.5, bargroupgap=0.12)
    lay['yaxis']['rangemode'] = 'normal'
    lay['yaxis']['zeroline'] = True
    return dict(data=data, layout=lay)


# ------------------------------------------------------------ сценарий
def scenario_time(s):
    x = _d(s['dates'])
    data = [_line(x, s['d_liq'].sum(axis=1), 'Δ жидкости', AQUA),
            _line(x, s['d_oil'].sum(axis=1), 'Δ нефти', ORANGE),
            _line(x, s['d_cm'].sum(axis=1), 'Δ жидкости по CM без калибровки', MAGENTA, 'dash')]
    lay = layout(y_title='изменение дебита, м³/сут')
    lay['yaxis']['rangemode'] = 'normal'
    lay['yaxis']['zeroline'] = True
    return dict(data=data, layout=lay)


def scenario_wells(m, s, top=12):
    dl, do = s['d_liq'][-1], s['d_oil'][-1]
    order = [j for j in np.argsort(-np.abs(dl))[:top] if abs(dl[j]) > 1e-6][::-1]
    if not order:
        return empty('Изменение закачки не затрагивает работающие добывающие скважины')
    names = [str(m.ds.prods[j]) for j in order]

    def bar(v, name, color):
        return dict(type='bar', orientation='h', y=names, x=_f(v[order]), name=name,
                    marker=dict(color=color, cornerradius=3, line=dict(color=SURFACE, width=1)),
                    hovertemplate='%{x:+.2f} м³/сут<extra>' + name + '</extra>')

    lay = layout(x_title='изменение дебита к концу периода, м³/сут', barmode='group', bargap=0.35,
                 bargroupgap=0.1, hovermode='y unified')
    lay['yaxis'].update(type='category', rangemode='normal', gridcolor='rgba(0,0,0,0)')
    lay['xaxis'].update(zeroline=True, zerolinecolor=AXIS)
    lay['margin']['l'] = 70
    return dict(data=[bar(dl, 'Δ жидкости', AQUA), bar(do, 'Δ нефти', ORANGE)], layout=lay)
