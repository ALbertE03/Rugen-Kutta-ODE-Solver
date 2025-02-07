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

            ecuaciones = []
            for i, cond in enumerate(cond_iniciales):
                derivada = expr.diff(x, i)
                ecuaciones.append(derivada.subs(x, 0) - cond)

            constantes = sp.solve(ecuaciones, C_symbols)

            sol_particular = expr.subs(constantes)
            # sol_particular_simplificada = sp.simplify(sol_particular)
            f_numeric_simplificado = sp.lambdify(x, sol_particular, "numpy")
            return (
                sp.latex(sol_homogenea),
                f_numeric_simplificado,
            )

        else:

            rhs = array[-1]
            ode_no_homogenea = sum(
                coef * y(x).diff(x, grado) for coef, grado in array[:-1]
            )

            sol_no_homogenea = sp.dsolve(sp.Eq(ode_no_homogenea, rhs), y(x))

            expr = sol_no_homogenea.rhs

            ecuaciones = []
            for i, cond in enumerate(cond_iniciales):

                derivada = expr.diff(x, i)
                ecuaciones.append(derivada.subs(x, 0) - cond)

            constantes = sp.solve(ecuaciones, C_symbols)

            sol_particular = expr.subs(constantes)
            sol_particular_simplificada = sp.simplify(sol_particular)
            f_numeric_simplificado = sp.lambdify(
                x, sol_particular_simplificada, "numpy"
            )
            return (
                sp.latex(sol_particular_simplificada),
                f_numeric_simplificado,
            )
