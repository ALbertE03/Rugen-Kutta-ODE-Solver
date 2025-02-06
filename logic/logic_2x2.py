import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp

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
        solutions = []

        if A[0][1] == 0 and A[1][1] == 0:
            new_A = A
            r = []
            for i in range(len(new_A)):
                aux = []
                for j in range(len(new_A)):
                    if i == j:
                        aux.append(f"e^{{{self.format_number(new_A[i][j])}t}}")
                    else:
                        aux.append("0")
                r.append(aux)
            mat_str = self.format_latex_string_matrix(r)
            solutions.append(f"x(t) = {mat_str} c")
            return solutions

        if A[0][0] == A[1][1] and A[0][1] + A[1][0] == 0:
            a_str = self.format_number(A[0][0], 2)
            b_str = self.format_number(abs(A[1][0]), 2)
            b = A[1][0]
            x1 = (
                rf"x_1(t) = e^{{({a_str})t}}\bigl(c_1\cos({b_str}t) "
                + rf"{'-' if b>0 else '+'} c_2\sin({b_str}t)\bigr)"
            )
            x2 = (
                rf"x_2(t) = e^{{({a_str})t}}\bigl(c_1\sin({b_str}t) "
                + rf"{'+' if b>0 else '-'} c_2\cos({b_str}t)\bigr)"
            )
            solutions.append(x1)
            solutions.append(x2)
            return solutions

        # Valores Propios Repetidos 
        if abs(eigenvalues[0] - eigenvalues[1]) < 1e-14:
            lambda_val = eigenvalues[0]
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except np.linalg.LinAlgError:
                P_inv = None
            rank_ev = np.linalg.matrix_rank(eigenvectors)
            if rank_ev < 2:
                
                if P_inv is not None:
                    t = sp.Symbol("t", real=True)
                    exp_lt = sp.exp(lambda_val * t)
                    J_exp = sp.Matrix([
                        [exp_lt, t*exp_lt],
                        [0,      exp_lt]
                    ])
                    P_sym = sp.Matrix(eigenvectors)
                    P_inv_sym = sp.Matrix(P_inv)
                    full_exp = P_sym * J_exp * P_inv_sym
                    full_exp_approx = full_exp.evalf(3)
                    latex_str = sp.latex(full_exp_approx, mat_delim="(", mat_str="pmatrix")
                    solutions.append(f"$$ x(t) = {latex_str} \\, c $$")
                    return solutions
                else:
                    solutions.append("$$ x(t) = P e^{J t} P^{-1} c, \\text{ no invertible }P. $$")
                    return solutions
            else:
                
                try:
                    P_inv = np.linalg.inv(eigenvectors)
                except np.linalg.LinAlgError:
                    P_inv = None
                if P_inv is not None:
                    t = sp.Symbol("t", real=True)
                    exp_lt = sp.exp(lambda_val * t)
                    D_exp = sp.diag(exp_lt, exp_lt)
                    P_sym = sp.Matrix(eigenvectors)
                    P_inv_sym = sp.Matrix(P_inv)
                    full_exp = P_sym * D_exp * P_inv_sym
                    full_exp_approx = full_exp.evalf(3)
                    latex_str = sp.latex(full_exp_approx, mat_delim="(", mat_str="pmatrix")
                    solutions.append(f"$$ x(t) = {latex_str} \\, c $$")
                    return solutions
                else:
                    solutions.append("$$ x(t) = P e^{D t} P^{-1} c, \\text{ no invertible }P. $$")
                    return solutions

        if abs(eigenvalues[0].imag) < 1e-20 and abs(eigenvalues[1].imag) < 1e-20:
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except:
                P_inv = "P^{-1}"
            if not isinstance(P_inv, str):
                new_A = np.einsum("ij,jk,kl->il", eigenvectors, A, P_inv)
                r = []
                for i in range(len(new_A)):
                    aux = []
                    for j in range(len(new_A)):
                        if i == j:
                            val = self.format_complex(new_A[i][j], 2)
                            aux.append(f"e^{{({val})t}}")
                        else:
                            aux.append("0")
                    r.append(aux)
                P_str = self.format_latex_complex_matrix(eigenvectors)
                r_str = self.format_latex_string_matrix(r)
                P_inv_str = self.format_latex_complex_matrix(P_inv)
                solutions.append(f"x(t) = {P_str} {r_str} {P_inv_str} c")
                return solutions
            else:
                solutions.append(
                    f"{eigenvectors}{A}{P_inv}c, ocurrió un error al calcular la inversa."
                )
                return solutions
        else:
            try:
                P = eigenvectors
                P_inv = np.linalg.inv(P)
            except:
                P_inv = "P^{-1}"
            if not isinstance(P_inv, str):
                Sol = np.einsum("ij,jk,kl->il", eigenvectors, A, P_inv)
                aux = [
                    [
                        f"e^{{({self.format_complex(Sol[0][0])})t}}cos({self.format_complex(Sol[0][1])}t)",
                        f"-e^{{({self.format_complex(Sol[0][0])})t}}sen({self.format_complex(Sol[0][1])}t)",
                    ],
                    [
                        f"e^{{({self.format_complex(Sol[1][0])})t}}sen({self.format_complex(Sol[1][1])}t)",
                        f"e^{{({self.format_complex(Sol[1][0])})t}}cos({self.format_complex(Sol[1][1])}t)",
                    ],
                ]
                P_str = self.format_latex_complex_matrix(P)
                aux_str = self.format_latex_string_matrix(aux)
                P_inv_str = self.format_latex_complex_matrix(P_inv)
                solutions.append(f"x(t) = {P_str} {aux_str} {P_inv_str} c")
                return solutions
            else:
                solutions.append(
                    f"{eigenvectors}{A}{P_inv}c, no se pudo invertir P."
                )
                return solutions

        return solutions

    def system_2x2(self, t, Y, A):
        return A @ Y

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
