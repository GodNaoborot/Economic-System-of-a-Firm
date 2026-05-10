import numpy as np
import scipy as sp
from scipy.optimize import fsolve

def euler_explicit(x, y0, f, **kwargs):
    """
    Явный метод Эйлера для систем ОДУ.
    
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
    dim = len(y0)
    y = np.zeros((n, dim))
    y[0] = y0

    for i in range(n - 1):
        h = x[i + 1] - x[i]
        y[i + 1] = y[i] + h * f(x[i], y[i], **kwargs)
    return y

def predictor_corrector(x, y0, f, **kwargs):
    """
    Метод предиктор-корректор для систем ОДУ.
    
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
    dim = len(y0)
    y = np.zeros((n, dim))
    y[0] = y0

    for i in range(n - 1):
        h = x[i + 1] - x[i]
        k1 = f(x[i], y[i], **kwargs)
        y_pred = y[i] + h * k1
        k2 = f(x[i] + h, y_pred, **kwargs)
        y[i + 1] = y[i] + h / 2.0 * (k1 + k2)
    return y


def euler_implicit(x, y0, f, **kwargs):
    """
    Неявный метод Эйлера для системы ОДУ.
    
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
    dim = len(y0)
    y = np.zeros((n, dim))
    y[0] = y0

    def equation(y_next, y_curr, h, x_next):
        return y_next - y_curr - h * f(x_next, y_next, **kwargs)

    for i in range(n - 1):
        h = x[i + 1] - x[i]
        y_guess = y[i]  # начальное приближение – значение с предыдущего шага
        y[i + 1] = fsolve(equation, y_guess, args=(y[i], h, x[i + 1]))
    return y

def rk4(x, y0, f, steps=3, **kwargs):
    """
    Метод Рунге-Кутты 4-го порядка для системы ОДУ.
    
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
    n_start = steps + 1
    m = len(y0)
    y = np.zeros((n_start, m))
    y[0] = y0
    for i in range(steps):
        h = x[i+1] - x[i]
        k1 = f(x[i], y[i], **kwargs)
        k2 = f(x[i] + h/2, y[i] + h/2 * k1, **kwargs)
        k3 = f(x[i] + h/2, y[i] + h/2 * k2, **kwargs)
        k4 = f(x[i] + h, y[i] + h * k3, **kwargs)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y