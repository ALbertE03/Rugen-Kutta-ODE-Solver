import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st


class Solve2x2:
    def __init__(self):
        pass

    def solve_system_2x2(self, A):
        eigenvalues, eigenvectors = np.linalg.eig(A)
        stable = all(np.real(eigenvalues) < 0)
        exp_At = expm(A)
        sol_y = np.random.rand(2, 1)
        return exp_At, stable, eigenvalues, eigenvectors, sol_y

    def format_matrix(self, matrix):
        return np.vectorize(
            lambda x: (
                f"{x:.2e}"
                if abs(x) >= 1e-7 or abs(x) < 1e-7
                else np.round(x, decimals=0)
            )
        )(matrix)

    def format_number(self, number):
        try:
            if np.abs(number) < 1e-7:
                return "0"
            elif number == int(number):
                return str(int(number))
            else:
                return f"{number:.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(number)

    def format_latex_matrix(self, matrix):
        formatted_matrix = "\\begin{pmatrix}"
        for row in matrix:
            formatted_row = " & ".join(self.format_number(val) for val in row)
            formatted_matrix += formatted_row + " \\\\ "
        formatted_matrix += "\\end{pmatrix}"
        return formatted_matrix

    def format_latex_matrix_precise(self, matrix):
        formatted_matrix = "\\begin{pmatrix}"
        for row in matrix:
            formatted_row = " & ".join(f"{val:.2e}" for val in row)
            formatted_matrix += formatted_row + " \\\\ "
        formatted_matrix += "\\end{pmatrix}"
        return formatted_matrix

    def format_special(self, matrix):
        def special_format(val):
            str_val = f"{val:.2e}"
            base, exponent = str_val.split("e")
            if exponent.endswith("00"):
                return base.split(".")[0]
            else:
                return "0"

        formatted_matrix = "\\begin{pmatrix}"
        for row in matrix:
            formatted_row = " & ".join(special_format(val) for val in row)
            formatted_matrix += formatted_row + " \\\\ "
        formatted_matrix += "\\end{pmatrix}"
        return formatted_matrix

        formatted_matrix = "\\begin{pmatrix}"
        for row in matrix:
            formatted_row = " & ".join(special_format(val) for val in row)
            formatted_matrix += formatted_row + " \\\\ "
        formatted_matrix += "\\end{pmatrix}"
        return formatted_matrix

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
            solutions.append(f"x_1(t) = {self.format_latex_matrix(r)}c")
            return solutions
        if A[0][0] == A[1][1] and A[0][1] + A[1][0] == 0:
            solutions.append(
                rf"x_1(t) = e^{{{self.format_number(-A[0][0])}t}}(c_1\cos({self.format_number(A[1][0])}t) - c_2\sin({self.format_number(A[1][0])}t))"
            )
            solutions.append(
                rf"x_2(t) = e^{{{self.format_number(-A[0][0])}t}}(c_1\sin({self.format_number(A[1][0])}t) + c_2\cos({self.format_number(A[1][0])}t))"
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
                    suma += f"+{N}^{i}t"
                s = f"x(t) = {self.format_latex_matrix(eigenvectors)} {self.format_latex_matrix(diag_end)} {self.format_special(P_inv)} (I + {self.format_latex_matrix(N)})c"
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
                print(P_inv)
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

            imagi1 = eigenvectors[0].imag[0]
            imagi2 = eigenvectors[1][0].imag

            real1 = eigenvectors[0].real[0]
            real2 = eigenvectors[1][0].real
            P = [
                [imagi1, real1],
                [imagi2, real2],
            ]
            try:
                P_inv = np.linalg.inv(P)
            except:
                P_inv = "P^{-1}"
            if not isinstance(P_inv, str):
                Sol = np.einsum("ij,jk,kl->il", P, A, P_inv)
                aux = [
                    [
                        f"e^({Sol[0][0]}t)cos(t)",
                        f"-e^({Sol[0][0]}t)sen(t)",
                    ],
                    [f"e^({Sol[0][0]}t)sen(t)"],
                    f"e^({Sol[0][0]}t)cos(t)",
                ]

                solutions.append(f"{P}{aux}{P_inv}c")
                return solutions

            return solutions.append(
                f"{P}{A}{P_inv}c, ocurrio un error al calcular la inversa y no se pudo seguir el procedimieto"
            )

    def system_2x2(self, t, Y, A):
        return A @ Y

    def plot_phase_diagram_2d(self, A):
        fig, ax = plt.subplots(figsize=(6, 6))
        t_span = [0, 10]
        t_eval = np.linspace(0, 10, 200)
        Y0_points = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
        ]

        for y0 in Y0_points:
            sol = solve_ivp(self.system_2x2, t_span, y0, args=(A,), t_eval=t_eval)
            ax.plot(sol.y[0], sol.y[1])
            ax.quiver(
                sol.y[0],
                sol.y[1],
                np.gradient(sol.y[0]),
                np.gradient(sol.y[1]),
                scale_units="xy",
                angles="xy",
                scale=1,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        st.pyplot(fig)
