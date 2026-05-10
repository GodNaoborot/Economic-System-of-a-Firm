import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_graph(x, y, name, **kwargs):
    """    
    Построение удобного графика (чтобы не писать постоянно плотли код)

    Параметры:
    ------------------------------
    - x: array-like, значения по оси X .
    - ys: array-like, значения по оси Y.
    - name: str, название кривой (легенда).
    - **layout_kwargs: параметры оформления графика (title, margin, и т.д.).
    ------------------------------
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name=name,
    ))
    fig.update_layout(**kwargs)
    return fig

import plotly.graph_objects as go

def plot_multiple(x, ys, names, **layout_kwargs):
    """    
    Построение графика для сравнения методов

    Параметры:
    ------------------------------
    - x: array-like, общие значения по оси X для всех кривых.
    - ys: list of array-like, список значений Y для каждой кривой.
    - names: list of str, имена кривых (легенда).
    - **layout_kwargs: параметры оформления графика (title, margin, и т.д.).
    ------------------------------
    """
    fig = go.Figure()
    for y, name in zip(ys, names):
        fig.add_trace(go.Scatter(x=x, y=y, name=name))
    fig.update_layout(**layout_kwargs, hovermode='x unified')
    return fig

def plot_lambda_grid(lambdas, x, y0, methods, f, accurate_method=None, **layout_kwargs):
    """
    Построение сетки графиков в зависимости от параметра \lambda

    Строит сетку графиков для разных значений параметра \lambda.

    Параметры:
    ------------------------------
    lambdas : list, Список значений параметра lambda (например, [-1, -5, -10, -50]).
    x : np.ndarray, Массив значений независимой переменной.
    y0 : float, Начальное условие y(x0) = y0.
    methods : dict, Словарь вида {"название метода": функция_метода}. Каждая функция должна принимать (x, y0, f, **params).
    f : callable, Правая часть уравнения f(x, y, **params).
    accurate_method : callable, optional, Функция точного решения (если есть). Должна принимать (x, y0, **params).
    **layout_kwargs : dict, Дополнительные параметры для layout всей фигуры (например, title, height, width).
    ------------------------------
    """
    n = len(lambdas)
    # Определяем число строк и столбцов для приблизительно квадратной сетки
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig = make_subplots(
        rows=rows, cols=cols,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=[f"λ = {lam}" for lam in lambdas]
    )

    for i, lam in enumerate(lambdas):
        row = i // cols + 1
        col = i % cols + 1

        params = {'alpha': lam}

        for name, method in methods.items():
            y = method(x, y0, f, **params)
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    name=name,
                    legendgroup=name,
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )

        if accurate_method is not None:
            y_acc = accurate_method(x, y0, **params)
            fig.add_trace(
                go.Scatter(x=x, y=y_acc, name='accurate',
                           line=dict(color='black', dash='dash'),
                           showlegend=(i == 0)),
                row=row, col=col
            )

    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    default_layout = dict(
        height=250 * rows,
        width=350 * cols,
        title=r"λ values influence ",
        hovermode='x unified'
    )
    default_layout.update(layout_kwargs)
    fig.update_layout(**default_layout)

    return fig


def plot_h_grid(h_list, bounds, y0, methods, f, lam, accurate_method=None, **layout_kwargs):
    """
    Построение сетки графиков в зависимости от шага h

    Параметры:
    ------------------------------
    h_list : list, Список значений шага h (например, [0.1, 0.05, 0.025, 0.01]).
    bounds : tuple (a, b), Границы интервала интегрирования.
    y0 : float, Начальное условие y(x0) = y0.
    methods : dict, Словарь вида {"название метода": функция_метода}. Каждая функция должна принимать (x, y0, f, **params).
    f : callable, Правая часть уравнения f(x, y, **params).
    lam : float, Значение параметра lambda (alpha) в уравнении.
    accurate_method : callable, optional, Функция точного решения (если есть). Должна принимать (x, y0, **params).
    **layout_kwargs : dict, Дополнительные параметры для layout всей фигуры (например, title, height, width).
    ------------------------------
    """
    n = len(h_list)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig = make_subplots(
        rows=rows, cols=cols,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=[f"h = {h}" for h in h_list]
    )

    legend_added = set()

    for i, h in enumerate(h_list):
        row = i // cols + 1
        col = i % cols + 1

        a, b = bounds
        n_points = int((b - a) / h) + 1
        x = np.linspace(a, b, n_points)

        if lam is not None:
            params = {'alpha': lam}
        else:
            params = {}

        for name, method in methods.items():
            y = method(x, y0, f, **params)
            show_legend = name not in legend_added
            if show_legend:
                legend_added.add(name)

            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    name=name,
                    legendgroup=name,
                    showlegend=show_legend
                ),
                row=row, col=col
            )

        if accurate_method is not None:
            x_dense = np.linspace(a, b, 500)
            y_acc = accurate_method(x_dense, y0, **params)
            acc_name = 'accurate'
            show_acc_legend = acc_name not in legend_added
            if show_acc_legend:
                legend_added.add(acc_name)

            fig.add_trace(
                go.Scatter(
                    x=x_dense, y=y_acc,
                    name=acc_name,
                    line=dict(color='black', dash='dash'),
                    legendgroup=acc_name,
                    showlegend=show_acc_legend
                ),
                row=row, col=col
            )

    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    default_layout = dict(
        height=250 * rows,
        width=350 * cols,
        title=f"Step h influence on numerical methods (λ = {lam})",
        hovermode='x unified'
    )
    default_layout.update(layout_kwargs)
    fig.update_layout(**default_layout)

    return fig

def plot_stability_perturbation(
    h: float,
    bounds: tuple,
    y0: float,
    methods: dict,
    f: callable,
    lam: float,
    epsilon: float = 1e-3,
    accurate_method: callable = None,
    **layout_kwargs
):
    """
    Построение графика, внося шум в начальное условие

    Параметры:
    ------------------------------
    h : float - шаг интегрирования
    bounds : tuple (a, b) - интервал
    y0 : float - начальное условие
    methods : dict - {"название": функция_метода}
    f : callable - правая часть f(x, y, **params)
    lam : float - параметр lambda (alpha)
    epsilon : float - величина возмущения
    accurate_method : callable, optional - точное решение
    **layout_kwargs - дополнительные параметры layout
    ------------------------------
    """
    a, b = bounds
    n = int((b - a) / h) + 1
    x = np.linspace(a, b, n)
    params = {'alpha': lam}

    n_methods = len(methods)
    subplot_titles = []
    for name in methods.keys():
        subplot_titles.append(f"{name}: solutions")
        subplot_titles.append(f"{name}: |difference| (log)")

    fig = make_subplots(
        rows=n_methods, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        subplot_titles=subplot_titles,
        column_widths=[0.6, 0.4]
    )

    legend_added = set()

    for i, (name, method) in enumerate(methods.items(), start=1):
        y_unpert = method(x, y0, f, **params)
        y_pert = method(x, y0 + epsilon, f, **params)
        diff = np.abs(y_pert - y_unpert)

        # Левый столбец: решения
        show_unpert = 'unperturbed' not in legend_added
        if show_unpert:
            legend_added.add('unperturbed')
        fig.add_trace(
            go.Scatter(x=x, y=y_unpert, name='unperturbed',
                       line=dict(color='blue'), legendgroup='unperturbed',
                       showlegend=show_unpert),
            row=i, col=1
        )

        show_pert = 'perturbed' not in legend_added
        if show_pert:
            legend_added.add('perturbed')
        fig.add_trace(
            go.Scatter(x=x, y=y_pert, name=f'perturbed (ε={epsilon})',
                       line=dict(color='red', dash='dot'), legendgroup='perturbed',
                       showlegend=show_pert),
            row=i, col=1
        )

        if accurate_method is not None:
            y_exact = accurate_method(x, y0, **params)
            show_exact = 'exact' not in legend_added
            if show_exact:
                legend_added.add('exact')
            fig.add_trace(
                go.Scatter(x=x, y=y_exact, name='exact',
                           line=dict(color='black', dash='dash'), legendgroup='exact',
                           showlegend=show_exact),
                row=i, col=1
            )

        # Правый столбец: разность (лог)
        diff_safe = diff + 1e-16
        show_diff = f'diff_{name}' not in legend_added
        if show_diff:
            legend_added.add(f'diff_{name}')
        fig.add_trace(
            go.Scatter(x=x, y=diff_safe, name=f'|diff| ({name})',
                       line=dict(color='green'), legendgroup=f'diff_{name}',
                       showlegend=show_diff),
            row=i, col=2
        )
        fig.update_yaxes(type="log", row=i, col=2, title_text="log(|diff|)")

    fig.update_xaxes(title_text="x", row=n_methods, col=1)
    fig.update_xaxes(title_text="x", row=n_methods, col=2)

    default_layout = dict(
        height=300 * n_methods,
        width=1000,
        title=f"Stability using perturbations (λ={lam}, h={h})",
        hovermode='x unified'
    )
    default_layout.update(layout_kwargs)
    fig.update_layout(**default_layout)

    return fig