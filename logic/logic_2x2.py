import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp
import warnings

warnings.simplefilter("error", RuntimeWarning)


class Solve2x2:
    def __init__(self):
        pass

    def solve_system_2x2(self, A):
        eigenvalues, eigenvectors = np.linalg.eig(A)
        stable = all(np.real(eigenvalues) < 0)
        exp_At = expm(A)
        sol_y = np.random.rand(2, 1)
        return exp_At, stable, eigenvalues, eigenvectors, sol_y

    def format_number(self, number, decimals=2):
        try:
            rounded = round(float(number), decimals)
            if abs(rounded) < 1e-14:
                return "0"
            if rounded.is_integer():
                return str(int(rounded))
            txt = f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")
            return txt
        except (TypeError, ValueError):
            return str(number)

    def format_complex(self, z: complex, decimals=2):
        if not isinstance(z, complex):
            return self.format_number(z, decimals)
        real_part = round(z.real, decimals)
        imag_part = round(z.imag, decimals)
        if abs(real_part) < 1e-14:
            real_part = 0
        if abs(imag_part) < 1e-14:
            imag_part = 0
        if imag_part == 0:
            return self.format_number(real_part, decimals)
        elif real_part == 0:
            return f"{self.format_number(imag_part, decimals)}i"
        elif imag_part > 0:
            return f"{self.format_number(real_part, decimals)} + {self.format_number(imag_part, decimals)}i"
        else:
            return f"{self.format_number(real_part, decimals)} - {self.format_number(-imag_part, decimals)}i"

    def format_latex_complex_matrix(self, M: np.ndarray, decimals=2):
        rows_str = []
        for row in M:
            row_items = [self.format_complex(val, decimals) for val in row]
            rows_str.append(" & ".join(row_items))
        matrix_str = "\\begin{pmatrix}" + " \\\\ ".join(rows_str) + "\\end{pmatrix}"
        return matrix_str

    def format_latex_string_matrix(self, M: list[list[str]]):
        rows_str = []
        for row in M:
            rows_str.append(" & ".join(row))
        matrix_str = "\\begin{pmatrix}" + " \\\\ ".join(rows_str) + "\\end{pmatrix}"
        return matrix_str

    def get_solutions_2x2(self, eigenvalues, eigenvectors, A):
        t = sp.symbols("t")
        x = sp.Function("x")(t)
        y = sp.Function("y")(t)
        z = sp.Function("z")(t)

        x_p = x.diff(t)
        y_p = y.diff(t)
        z_p = z.diff(t)
        A_2x2 = sp.Matrix(A)
        f = sp.Matrix([x, y])
        f_p = sp.Matrix([x_p, y_p])

        eqs = f_p - A_2x2 * f
        eq1 = sp.Eq(eqs[0], 0)
        eq2 = sp.Eq(eqs[1], 0)
        sol = sp.dsolve([eq1, eq2], [x, y])
        return sol

    def system_2x2(self, t, Y, A):
        try:
            return A @ Y
        except:
            raise RuntimeWarning("")

    def plot_phase_diagram_2d(self, A) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        t_span = [0, 10]
        t_eval = np.linspace(0, 10, 200)
        Y0_points = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
        ]
        colors = ["b", "g", "r", "orange"]
        for idx, y0 in enumerate(Y0_points):
            sol = solve_ivp(self.system_2x2, t_span, y0, args=(A,), t_eval=t_eval)
            ax.plot(sol.y[0], sol.y[1], color=colors[idx], alpha=0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        st.pyplot(fig)
