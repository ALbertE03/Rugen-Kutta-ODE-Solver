import sympy as sp


def solve(array, HOMOGENIA=False):
    x = sp.symbols("x")
    y = sp.Function("y")

    if HOMOGENIA:

        ode_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array)
        sol_homogenea = sp.dsolve(sp.Eq(ode_homogenea, 0), y(x))

        sp.pprint(sol_homogenea)

    else:

        rhs = array[-1]
        ode_no_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array[:-1])

        sol_no_homogenea = sp.dsolve(sp.Eq(ode_no_homogenea, rhs), y(x))

        sp.pprint(sol_no_homogenea)


solve([(2, 2), (1, 1)], True)

solve([(2, 2), (1, 1), sp.sin(sp.symbols("x"))], False)
