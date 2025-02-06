import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class Sup:
    def __init__(self):
        pass

    def plot_homog(self):
        pass

    def plot_no_homog(self):
        pass

    def get_solution(self, array, cond_iniciales, HOMOG=True):
        x = sp.Symbol("x")
        y = sp.Function("y")
        C_symbols = sp.symbols(f"C1:{len(cond_iniciales) + 1}")
        if HOMOG:
            ode_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array)
            sol_homogenea = sp.dsolve(sp.Eq(ode_homogenea, 0), y(x))
            expr = sol_homogenea.rhs
            for i, cond in enumerate(cond_iniciales):
                expr = expr.subs(C_symbols[i], cond)
            expr_eval = expr.evalf(3)
            f_numeric = sp.lambdify(x, expr, "numpy")
            return sp.latex(expr_eval), f_numeric
        rhs = array[-1]
        ode_no_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array[:-1])
        sol_no_homogenea = sp.dsolve(sp.Eq(ode_no_homogenea, rhs), y(x))
        expr = sol_no_homogenea.rhs
        for i, cond in enumerate(cond_iniciales):
            expr = expr.subs(C_symbols[i], cond)
        expr_eval = expr.evalf(3)
        f_numeric = sp.lambdify(x, expr, "numpy")
        return sp.latex(expr_eval), f_numeric
