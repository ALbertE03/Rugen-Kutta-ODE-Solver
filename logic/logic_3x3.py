import numpy as np
import sympy as sp
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

class Solve3x3:
    def __init__(self):
        pass

    def matrix_exponential(self, A, t):
        n = A.shape[0]
        exp_At = np.eye(n) + A*t
        for k in range(2, 20):
            exp_At += np.linalg.matrix_power(A*t, k) / np.math.factorial(k)
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
        solutions = []
        unique_vals, counts = np.unique(eigenvalues, return_counts=True)
        count_eigenvalues = dict(zip(unique_vals, counts))
        is_complex = any(np.iscomplex(ev) and abs(ev.imag) > 1e-14 for ev in eigenvalues)
        t = sp.Symbol("t", real=True)

        # --- Complex eigenvalues case ---
        if is_complex:
            complex_eigs = [ev for ev in eigenvalues if np.iscomplex(ev) and abs(ev.imag) > 1e-14]
            real_eigs = [ev for ev in eigenvalues if abs(ev.imag) < 1e-14]
            if len(real_eigs) > 0:
                real_val = round(real_eigs[0].real, 3)
            else:
                real_val = 0.0
            cplx = complex_eigs[0]
            alpha = round(cplx.real, 3)
            beta = round(cplx.imag, 3)
            v = np.array(eigenvectors).flatten()
            v1 = sp.Matrix(v[0:3].real)
            v2 = sp.Matrix(v[3:6].real)
            v3 = sp.Matrix(v[6:9].real)
            c1, c2, c3 = sp.symbols("c1 c2 c3", real=True)
            part1 = c1*sp.exp(real_val*t)*v1
            part2 = sp.exp(alpha*t)*(c2*sp.cos(beta*t)*v2 + c3*sp.sin(beta*t)*v3)
            total_solution = (part1 + part2).evalf(3)
            latex_expr = sp.latex(total_solution, mat_delim="(", mat_str="pmatrix")
            solutions.append(f"$$ x(t) = {latex_expr} $$")
            return solutions

        # --- Distinct real eigenvalues case ---
        if len(count_eigenvalues) == 3:
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except np.linalg.LinAlgError:
                P_inv = None

            if P_inv is not None:
                spP = sp.Matrix(eigenvectors).evalf(3)
                spPinv = sp.Matrix(P_inv).evalf(3)
                lam1 = round(eigenvalues[0].real, 3)
                lam2 = round(eigenvalues[1].real, 3)
                lam3 = round(eigenvalues[2].real, 3)
                exp_diag = sp.diag(
                    sp.exp(lam1*t),
                    sp.exp(lam2*t),
                    sp.exp(lam3*t)
                ).evalf(3)
                p_str = sp.latex(spP, mat_delim="(", mat_str="pmatrix")
                p_inv_str = sp.latex(spPinv, mat_delim="(", mat_str="pmatrix")
                exp_diag_str = sp.latex(exp_diag, mat_delim="(", mat_str="pmatrix")
                solutions.append(f"$$ x(t) = {p_str} {exp_diag_str} {p_inv_str} c $$")
                return solutions
            else:
                solutions.append("$$ x(t) = P\\,e^{D t}\\,P^{-1}c,\\text{ pero }P\\text{ no es invertible.} $$")
                return solutions

        # --- Repeated eigenvalues (Jordan-block) case ---
        else:
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except np.linalg.LinAlgError:
                P_inv = None
            if P_inv is not None:
                diag_l = np.diag([round(ev.real, 3) for ev in eigenvalues])
                S = eigenvectors @ diag_l @ P_inv
                N = A - S
                k = 1
                while np.allclose(N, 0):
                    N = N @ N
                    k += 1
                    if k >= 3:
                        break
                sum_str = ""
                for i_power in range(1, k + 1):
                    if i_power == 1:
                        sum_str += f"+(N^{i_power} t^{i_power})"
                    else:
                        fac = np.math.factorial(i_power)
                        sum_str += f"+\\frac{{N^{i_power} t^{i_power}}}{{{fac}}}"
                solutions.append(
                    f"$$ x(t) = P\\,e^{{(\\mathrm{{diag}}(\\lambda_i)) t}}\\,P^{{-1}}\\bigl(I {sum_str}\\bigr)c $$"
                )
                return solutions
            else:
                solutions.append("$$ \\text{Error: no se pudo invertir }P\\text{.} $$")
                return solutions

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
            title="Diagrama de Fase en 3D",
        )
        st.plotly_chart(fig)
