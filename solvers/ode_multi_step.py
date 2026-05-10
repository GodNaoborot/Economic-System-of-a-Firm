import numpy as np
from collections import deque
from scipy.optimize import fsolve

def adams_bashforth_4(x, y0, f, **kwargs):
    """
    Явный метод Адамса-Бешфорта 4-го порядка для системы ОДУ.

    Параметры:
    ----------
    x : np.ndarray
        Сетка значений x из [a, b]
    y0 : np.ndarray
        Начальное условие (вектор)
    f : callable
        Правая часть f(x, y, **kwargs), возвращает np.ndarray той же размерности, что y
    **kwargs : dict
        Дополнительные параметры для f
    """
    n = len(x)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0

    # начальные 3 шага — RK4
    for i in range(3):
        h = x[i+1] - x[i]
        k1 = f(x[i], y[i], **kwargs)
        k2 = f(x[i] + h/2, y[i] + h/2 * k1, **kwargs)
        k3 = f(x[i] + h/2, y[i] + h/2 * k2, **kwargs)
        k4 = f(x[i] + h, y[i] + h * k3, **kwargs)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    # юзаем deque чтоб эффективнее работать с данными, т.к может быть переполнение памяти в теории
    f_vals = deque([f(x[i], y[i], **kwargs) for i in range(4)], 4)
    for i in range(3, n-1):
        h = x[i+1] - x[i]
        y[i+1] = y[i] + h/24 * (55*f_vals[-1] - 59*f_vals[-2] + 37*f_vals[-3] - 9*f_vals[-4])
        f_vals.append(f(x[i+1], y[i+1], **kwargs))
    return y

def adams_moulton_4(x, y0, f, **kwargs):
    """
    Неявный метод Адамса-Моултона 4-го порядка для системы ОДУ

    Параметры:
    ----------
    x : np.ndarray
        Сетка значений x из [a, b]
    y0 : np.ndarray
        Начальное условие (вектор)
    f : callable
        Правая часть f(x, y, **kwargs), возвращает np.ndarray той же размерности, что y
    **kwargs : dict
        Дополнительные параметры для f
    """
    n = len(x)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0

    # начальные 3 шага — RK4
    for i in range(3):
        h = x[i+1] - x[i]
        k1 = f(x[i], y[i], **kwargs)
        k2 = f(x[i] + h/2, y[i] + h/2 * k1, **kwargs)
        k3 = f(x[i] + h/2, y[i] + h/2 * k2, **kwargs)
        k4 = f(x[i] + h, y[i] + h * k3, **kwargs)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    for i in range(3, n-1):
        h = x[i+1] - x[i]
        f_n   = f(x[i],   y[i],   **kwargs)
        f_n1  = f(x[i-1], y[i-1], **kwargs)
        f_n2  = f(x[i-2], y[i-2], **kwargs)

        def equation(yn1):
            return yn1 - y[i] - h/24 * (9*f(x[i+1], yn1, **kwargs) + 19*f_n - 5*f_n1 + 1*f_n2)

        y_guess = y[i]
        y[i+1] = fsolve(equation, y_guess)
    return y

def adams_bashforth_moulton_4(x, y0, f, **kwargs):
    """
    Предиктор-корректор (Адамса-Башфорта-Моултона) 4-го порядка для системы ОДУ.

    Параметры:
    ----------
    x : np.ndarray
        Сетка значений x из [a, b]
    y0 : np.ndarray
        Начальное условие (вектор)
    f : callable
        Правая часть f(x, y, **kwargs), возвращает np.ndarray той же размерности, что y
    **kwargs : dict
        Дополнительные параметры для f
    """
    n = len(x)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0

    # начальные 3 шага — RK4
    for i in range(3):
        h = x[i+1] - x[i]
        k1 = f(x[i], y[i], **kwargs)
        k2 = f(x[i] + h/2, y[i] + h/2 * k1, **kwargs)
        k3 = f(x[i] + h/2, y[i] + h/2 * k2, **kwargs)
        k4 = f(x[i] + h, y[i] + h * k3, **kwargs)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    

    # юзаем deque чтоб эффективнее работать с данными, т.к может быть переполнение памяти в теории
    f_vals = deque([f(x[i], y[i], **kwargs) for i in range(4)], 4)
    for i in range(3, n-1):
        h = x[i+1] - x[i]
        # предиктор (Адамс-Бешфорт)
        y_pred = y[i] + h/24 * (55*f_vals[-1] - 59*f_vals[-2] + 37*f_vals[-3] - 9*f_vals[-4])
        # корректор (Адамс-Моултон)
        f_pred = f(x[i+1], y_pred, **kwargs)
        y_corr = y[i] + h/24 * (9*f_pred + 19*f_vals[-1] - 5*f_vals[-2] + f_vals[-3])
        y[i+1] = y_corr
        f_vals.append(f(x[i+1], y[i+1], **kwargs))
    return y

def bdf4_system(x, y0, f, **kwargs):
    """
    BDF4 (Гира 4-го порядка) для системы ОДУ.

    Параметры:
    ----------
    x : np.ndarray
        Сетка значений x из [a, b]
    y0 : np.ndarray
        Начальное условие (вектор)
    f : callable
        Правая часть f(x, y, **kwargs), возвращает np.ndarray той же размерности, что y
    **kwargs : dict
        Дополнительные параметры для f
    """
    n = len(x)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0

    # начальные 3 шага — RK4
    for i in range(3):
        h = x[i+1] - x[i]
        k1 = f(x[i], y[i], **kwargs)
        k2 = f(x[i] + h/2, y[i] + h/2 * k1, **kwargs)
        k3 = f(x[i] + h/2, y[i] + h/2 * k2, **kwargs)
        k4 = f(x[i] + h, y[i] + h * k3, **kwargs)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    for i in range(3, n-1):
        h = x[i+1] - x[i]
        def eq(yn1):
            return (yn1
                    - (48/25)*y[i]
                    + (36/25)*y[i-1]
                    - (16/25)*y[i-2]
                    + (3/25)*y[i-3]
                    - (12/25)*h * f(x[i+1], yn1, **kwargs))
        
        y_guess = y[i]
        y[i+1] = fsolve(eq, y_guess)
    return y