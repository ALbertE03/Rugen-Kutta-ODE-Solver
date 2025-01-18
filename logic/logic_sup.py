import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


class Sup:
    def __init__(self):
        pass

    def solve_homogenia():
        pass

    def get_solution(self, array, HOMOG=True):
        x = sp.symbols("x")
        y = sp.Function("y")

        if HOMOG:

            ode_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array)
            sol_homogenea = sp.dsolve(sp.Eq(ode_homogenea, 0), y(x))

            return sp.latex(sol_homogenea)

        rhs = array[-1]
        ode_no_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array[:-1])

        sol_no_homogenea = sp.dsolve(sp.Eq(ode_no_homogenea, rhs), y(x))

        return sp.latex(sol_no_homogenea)


# ejemplo
sup = Sup()
print(sup.get_solution([(2, 2), (1, 1)]))
# tienes que limitar las opciones de las no homogenias, xq necesitas pasarlas simbolicamete con sympy
print(sup.get_solution([(2, 2), (1, 1), sp.sin(sp.symbols("x"))], False))
