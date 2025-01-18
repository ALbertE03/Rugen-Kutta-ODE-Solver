import sympy


def solve(array, HOMOGENIA=False):
    x = sp.symbols("x")
    y = sp.Function("y")
    # array es lista de tuplas de (coeficiente, grado de la derivada), si HOMOGENIA está true es que el ultimo elemento es el miembro derecho
    if HOMOGENIA:
        ode_homogenea = sum(coef * y(x).diff(x, grado) for coef, grado in array)
        sol_homogenea = sp.dsolve(sp.Eq(ode_homogenea, 0), y(x))

        return sol_homogenea

    ode_no_homogenea = sum(i[0] * y(x).diff(x, i[1]) for i in range(len(array) - 2))

    rhs = array[-1]
    sol_no_homogenea = sp.dsolve(sp.Eq(ode_no_homogenea, rhs), y(x))

    return sol_no_homogenea
