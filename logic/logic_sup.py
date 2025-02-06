import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from logic.error import Len


class Sup:
    def __init__(self) -> None:
        pass

    def plot_homog(self) -> None:
        pass

    def plot_no_homog(self) -> None:
        pass

    def get_solution(
        self,
        array: list[tuple[int | float]],
        cond_iniciales: list[int | float],
        HOMOG=True,
    ) -> tuple[str, callable]:

        x = sp.symbols("x")
        y = sp.Function("y")
        C_symbols = sp.symbols(f"C1:{len(cond_iniciales) + 1}")
        if HOMOG:

            ode_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array)
            sol_homogenea = sp.dsolve(sp.Eq(ode_homogenea, 0), y(x))

            expr = sol_homogenea.rhs
            for i, cond in enumerate(cond_iniciales):
                expr = expr.subs(C_symbols[i], cond)

            f_numeric = sp.lambdify(x, expr, "numpy")

            return (
                sp.latex(sol_homogenea),
                f_numeric,
            )  

        rhs = array[-1]
        ode_no_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array[:-1])

        sol_no_homogenea = sp.dsolve(sp.Eq(ode_no_homogenea, rhs), y(x))
        expr = sol_no_homogenea.rhs
        for i, cond in enumerate(cond_iniciales):
            expr = expr.subs(C_symbols[i], cond)
        f_numeric = sp.lambdify(x, expr, "numpy")

        return (
            sp.latex(sol_no_homogenea),
            f_numeric,
        )  



