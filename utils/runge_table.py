import numpy as np
import pandas as pd

def runge_table(
        bounds: np.array,
        steps: np.array,
        y0: float,
        method: callable,
        f: callable,
        name = None,
        exact_solution=None,
        etalon_solution=None, 
        **params
) -> pd.DataFrame:
    """
    Таблица Рунге для порядка и ошибки методов и их сравнения

    Параметры:
    ------------------------------
    bounds : list(float, float) - границы на котором считается решение
    steps : float - шаг сетки
    y0 : float - начальное условие
    f : callable - функция в правой части
    name : str - название у метода в таблице
    params : dict - параметры для методов решения
    ------------------------------
    """
    a, b = bounds
    rows = []
    for h in steps:

        n_h   = int((b - a) / h) + 1
        n_2h  = int((b - a) / (2 * h)) + 1
        n_4h  = int((b - a) / (4 * h)) + 1

        x_h = np.linspace(a, b, n_h)
        x_2h = np.linspace(a, b, n_2h)
        x_4h = np.linspace(a, b, n_4h)

        y_h = method(x_h, y0, f, **params)
        y_2h = method(x_2h, y0, f, **params)
        y_4h = method(x_4h, y0, f, **params)

        # ошибка по двум шагам
        eps_h = np.max(np.abs(y_h[::2] - y_2h))
        eps_2h = np.max(np.abs(y_2h[::2] - y_4h))

        # порядок по двум шагам
        p = np.log2(eps_2h / eps_h) if eps_h > 0 else np.nan

        err_h = 1 / (2**p - 1) * eps_h

        row = [h, err_h, p]

        # ошибка и порядок относительно точного решения
        if exact_solution is not None:
            y_ex = exact_solution(x_h, y0, **params)
            err_exact = np.max(np.abs(y_h - y_ex))

            y_ex_2h = exact_solution(x_2h, y0, **params)
            err_exact_2h = np.max(np.abs(y_2h - y_ex_2h))
            p_exact = np.log2(err_exact_2h / err_exact) if err_exact > 0 else np.nan
            
            row.extend([err_exact, p_exact])

        # ошибка относительно эталонного решения
        if etalon_solution is not None:
            y_et = etalon_solution(x_h, y0, f, **params)
            err_et = np.max(np.abs(y_h - y_et))
            row.append(err_et)

        rows.append(row)

    columns = ['Eps_h', 'Order_p']
    if exact_solution is not None:
        columns.extend(['Eps_h_exact', 'Order_p_exact'])
    if etalon_solution is not None:
        columns.append('Eps_h_etalon')

    runge_df = pd.DataFrame(data=rows, columns=['Step'] + columns)
    runge_df = runge_df.set_index('Step')

    if name is None:
        name = "Unknown Method"
        
    runge_df.columns = pd.MultiIndex.from_product([[name], columns])

    return runge_df


def runge_table_system(
    bounds: list[float, float],
    steps: np.ndarray,
    y0: np.ndarray,
    method: callable,
    f: callable,
    name: str = None,
    exact_solution=None,
    etalon_solution=None,
    **params
) -> pd.DataFrame:
    """
    Таблица Рунге для СИСТЕМ ОДУ.

    Параметры:
    ------------------------------
    bounds : list(float, float) - границы на котором считается решение
    steps : float - шаг сетки
    y0 : np.ndarray - вектор начальных условий
    f : callable - функция в правой части
    name : str - название у метода в таблице
    params : dict - параметры для методов решения
    ------------------------------
    """
    a, b = bounds
    rows = []
    
    for h in steps:
        n_h  = int((b - a) / h) + 1
        x_h  = np.linspace(a, b, n_h)
        
        n_2h = int((b - a) / (2 * h)) + 1
        x_2h = np.linspace(a, b, n_2h)
        
        n_4h = int((b - a) / (4 * h)) + 1
        x_4h = np.linspace(a, b, n_4h)

        y_h  = method(x_h,  y0, f, **params)
        y_2h = method(x_2h, y0, f, **params)
        y_4h = method(x_4h, y0, f, **params)

        # Выравниваем длины для сравнения
        min_len_2 = min(len(y_h[::2]), len(y_2h))
        min_len_4 = min(len(y_2h[::2]), len(y_4h))

        eps_h  = np.max(np.abs(y_h[::2][:min_len_2] - y_2h[:min_len_2]))
        eps_2h = np.max(np.abs(y_2h[::2][:min_len_4] - y_4h[:min_len_4]))

        p = np.log2(eps_2h / eps_h) if eps_h > 1e-16 else np.nan

        row = [h, eps_h, p]

        if exact_solution is not None:
            pass

        rows.append(row)

    df = pd.DataFrame(rows, columns=['Step', 'Eps_h', 'Order_p'])
    df = df.set_index('Step')

    if name is None:
        name = method.__name__
    df.columns = pd.MultiIndex.from_product([[name], df.columns])

    return df