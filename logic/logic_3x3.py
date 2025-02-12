import numpy as np
import sympy as sp
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import warnings

warnings.simplefilter("error", RuntimeWarning)


class Solve3x3:
    def __init__(self):
        pass

    def matrix_exponential(self, A, t):
        n = A.shape[0]
        exp_At = np.eye(n) + A * t
        for k in range(2, 20):
            exp_At += np.linalg.matrix_power(A * t, k) / np.math.factorial(k)
        return exp_At

    def system_3x3(self, t, Y, A):
        return A @ Y

    def solve_system_3x3(self, A, Y0):
        eigenvalues, eigenvectors = np.linalg.eig(A)
        stable = all(np.real(eigenvalues) < 0)
        exp_At = self.matrix_exponential(A, 1)
        t_span = [0, 10]
        t_eval = np.linspace(0, 10, 200)
        sol = solve_ivp(self.system_3x3, t_span, Y0, args=(A,), t_eval=t_eval)
        return exp_At, stable, eigenvalues, eigenvectors, sol.y

    def get_solutions_3x3(self, eigenvalues, eigenvectors, A):
        t = sp.symbols("t")
        x = sp.Function("x")(t)
        y = sp.Function("y")(t)
        z = sp.Function("z")(t)

        x_p = x.diff(t)
        y_p = y.diff(t)
        z_p = z.diff(t)
        A_3x3 = sp.Matrix(A)
        f = sp.Matrix([x, y, z])
        f_p = sp.Matrix([x_p, y_p, z_p])

        eqs = f_p - A_3x3 * f
        eq1 = sp.Eq(eqs[0], 0)
        eq2 = sp.Eq(eqs[1], 0)
        eq3 = sp.Eq(eqs[2], 0)
        sol = sp.dsolve([eq1, eq2, eq3], [x, y, z])
        sol_simplificada = [sp.simplify(so) for so in sol]

        return sol_simplificada

    def plot_phase_diagram_3d(self, A, sol_y):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=sol_y[0],
                y=sol_y[1],
                z=sol_y[2],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color="blue", width=2),
            )
        )
        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )
        st.plotly_chart(fig)
