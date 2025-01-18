import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp
from typing import Tuple


class Solve2x2:
    def __init__(self) -> None:
        pass

    def solve_system_2x2(self, A) -> Tuple:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        stable = all(np.real(eigenvalues) < 0)
        exp_At = expm(A)
        sol_y = np.random.rand(2, 1)
        return exp_At, stable, eigenvalues, eigenvectors, sol_y

    def format_matrix(self, matrix) -> np.vectorize:
        return np.vectorize(
            lambda x: (
                f"{x:.2e}"
                if abs(x) >= 1e-7 or abs(x) < 1e-7
                else np.round(x, decimals=0)
            )
        )(matrix)

    def format_number(self, number) -> str:
        try:
            if np.abs(number) < 1e-7:
                return "0"
            elif number == int(number):
                return str(int(number))
            else:
                return f"{number:.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(number)

    def format_latex_matrix(self, matrix) -> str:
        formatted_matrix = "\\begin{pmatrix}"
        for row in matrix:
            formatted_row = " & ".join(self.format_number(val) for val in row)
            formatted_matrix += formatted_row + " \\\\ "
        formatted_matrix += "\\end{pmatrix}"
        return formatted_matrix

    def format_latex_matrix_precise(self, matrix) -> str:
        formatted_matrix = "\\begin{pmatrix}"
        for row in matrix:
            formatted_row = " & ".join(f"{val:.2e}" for val in row)
            formatted_matrix += formatted_row + " \\\\ "
        formatted_matrix += "\\end{pmatrix}"
        return formatted_matrix

    def format_special(self, matrix) -> str:
        def special_format(val):
            str_val = f"{val:.2e}"
            base, exponent = str_val.split("e")
            if base.endswith("00"):
                return base.split(".")[0]
            elif base.endswith("50") or base.endswith("5"):
                return "0"
            else:
                return f"{float(base):.0f}"

        formatted_matrix = "\\begin{pmatrix}"
        for row in matrix:
            formatted_row = " & ".join(special_format(val) for val in row)
            formatted_matrix += formatted_row + " \\\\ "
        formatted_matrix += "\\end{pmatrix}"
        return formatted_matrix

    def get_solutions_2x2(self, eigenvalues, eigenvectors, A) -> list[str]:
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
            solutions.append(f"x_1(t) = {self.format_latex_matrix(r)}c")
            return solutions
        if A[0][0] == A[1][1] and A[0][1] + A[1][0] == 0:
            solutions.append(
                rf"x_1(t) = e^{{{self.format_number(A[0][0])}t}}(c_1\cos({self.format_number(A[1][0])}t) - c_2\sin({self.format_number(A[1][0])}t))"
            )
            solutions.append(
                rf"x_2(t) = e^{{{self.format_number(A[1][1])}t}}(c_1\sin({self.format_number(A[1][0])}t) + c_2\cos({self.format_number(A[1][0])}t))"
            )
            return solutions

        elif eigenvalues[0] == eigenvalues[1]:
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except:
                P_inv = "P^-1"

            diag_l = np.zeros((2, 2))
            np.fill_diagonal(diag_l, eigenvalues)
            if not isinstance(P_inv, str):
                S = np.einsum("ij,jk,kl->il", eigenvectors, diag_l, P_inv)
                N = np.subtract(A, S)
                k = 1
                while np.all(N == 0):
                    N = np.dot(N, N)
                    k += 1
                    if k >= eigenvectors.shape[0]:
                        break
                diag_end = []
                for i in range(len(diag_l)):
                    aux = []
                    for j in range(len(diag_l)):
                        if i == j:
                            aux.append(f"e^{{{self.format_number(diag_l[i][j])}t}}")
                        else:
                            aux.append("0")
                    diag_end.append(aux)
                suma = ""
                for i in range(1, k + 1):
                    f = np.math.factorial(i)
                    if i <= 1:
                        suma += f"+({N}^{i}t^{i})"
                    else:
                        suma += f"+({N}^{i}t^{i})/{np.math.factorial(i)}"
                s = f"x(t) = {self.format_latex_matrix(eigenvectors)} {self.format_latex_matrix(diag_end)} {self.format_special(P_inv)} (I + {self.format_latex_matrix(suma)})c"  # esta ultima matriz es la matriz suma, no N
                solutions.append(s)
                return solutions
            else:
                solutions.append(
                    f"{eigenvectors}{diag_l}{P_inv}c, ocurrió un error al calcular la inversa y no se puedo seguir el procedimieto"
                )
                return solutions
        elif abs(eigenvalues[0].imag) < 1e-20 and abs(eigenvalues[1].imag) < 1e-20:
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
                            aux.append(f"e^{{{self.format_number(new_A[i][j])}t}}")
                        else:
                            aux.append("0")
                    r.append(aux)
                solutions.append(
                    f"x(t) = {self.format_latex_matrix(eigenvectors)} {self.format_latex_matrix(r)} {self.format_special(P_inv)} c"
                )
                return solutions
            else:
                solutions.append(
                    f"{eigenvectors}{A}{P_inv}c, ocurrió un error al calcular la inversa y no se puedo seguir el procedimieto"
                )
                return solutions
        else:
            try:
                P_inv = np.linalg.inv(P)
            except:
                P_inv = "P^{-1}"
            if not isinstance(P_inv, str):
                Sol = np.einsum("ij,jk,kl->il", eigenvectors, A, P_inv)
                aux = [
                    [
                        f"e^({Sol[0][0]}t)cos({Sol[0][1]}t)",
                        f"-e^({Sol[0][0]}t)sen({Sol[0][1]}t)",
                    ],
                    [f"e^({Sol[0][0]}t)sen({Sol[0][1]}t)"],
                    f"e^({Sol[0][0]}t)cos({Sol[0][1]}t)",
                ]

                solutions.append(f"{P}{aux}{P_inv}c")
                return solutions

            return solutions.append(
                f"{P}{A}{P_inv}c, ocurrio un error al calcular la inversa y no se pudo seguir el procedimieto"
            )

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

        colors = [
            "b",
            "g",
            "r",
            "orange",
        ]  # Lista de colores para las curvas, incluyendo orange

        for idx, y0 in enumerate(Y0_points):
            sol = solve_ivp(self.system_2x2, t_span, y0, args=(A,), t_eval=t_eval)
            ax.plot(
                sol.y[0], sol.y[1], color=colors[idx], alpha=0.7
            )  # Curvas coloridas

            """for i in range(0, len(sol.y[0]), 3):  # Intervalo cambiado a 3
                x, y = sol.y[0][i], sol.y[1][i]
                dx, dy = np.gradient(sol.y[0])[i], np.gradient(sol.y[1])[i]
                angle = np.arctan2(dy, dx)  # Ángulo de la tangente a la curva
                symbol = ">"  # Símbolo de flecha sin cola
                distance_from_center = np.sqrt(x**2 + y**2)
                max_distance = np.sqrt(2)
                size = (
                    100 + 290 * (distance_from_center / max_distance) ** 3
                )  # Tamaño ajustado para disminuir más rápido
                # alpha = distance_from_center / max_distance  # Transparencia ajustada
                ax.text(
                    x,
                    y,
                    symbol,
                    fontsize=size / 10,
                    # color=(0, 0, 0, alpha),
                    ha="center",
                    va="center",
                    rotation=np.degrees(angle),
                )
        """
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        st.pyplot(fig)
