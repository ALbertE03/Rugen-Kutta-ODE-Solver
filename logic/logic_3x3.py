import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import streamlit as st
import sympy as sp


class Solve3x3:
    def __init__(self) -> None:
        pass

    def matrix_exponential(self, A, t) -> np.ndarray:
        n = A.shape[0]
        exp_At = np.eye(n) + A * t
        for k in range(2, 20):
            exp_At += np.linalg.matrix_power(A * t, k) / np.math.factorial(k)
        return exp_At

    def solve_system_3x3(self, A, Y0) -> tuple:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        stable = all(np.real(eigenvalues) < 0)
        exp_At = matrix_exponential(A, 1)

        t_span = [0, 10]
        t_eval = np.linspace(0, 10, 200)
        sol = solve_ivp(system_3x3, t_span, Y0, args=(A,), t_eval=t_eval)

        return exp_At, stable, eigenvalues, eigenvectors, sol.y

    def get_solutions_3x3(self, eigenvalues, eigenvectors, Y0) -> list[str]:
        solutions = []
        unique, count = np.unique(eigenvalues, return_counts=True)
        count_eigenvalues = dict(zip(unique, count))
        is_compjex = False
        index = -1
        complex_index = -1
        for j, i in enumerate(count_eigenvalues):
            if np.isreal(i):
                index = i
            if np.iscomplex(i):
                is_compjex = True
                complex_index = i
                break

        if (
            is_compjex
        ):  # esta parte tengo que probarla, xq necesito ver una cosa en la innterfaz cuando la hagas
            v = np.array(eigenvectors).flatten()
            part_real = complex_index.real
            part_imag = complex_index.imag
            part_imga_c = -complex_index.imag
            value_real = index
            c1, c2, c3 = sp.symbols("c1", "c2", "c3")

            v1 = sp.matrix(v[:2].real)
            v2 = sp.matrix(v[2:5].real)
            v3 = sp.matrix(v[5:].real)
            s = c1 * sp.expm(value_real * t) * v1
            s2 = sp.expm(part_real * t) * (
                c2 * sp.cos(part_imag * t) * v2 + c3 * sp.sin(part_imag * t) * v3
            )
            return sp.latex(s + s2)
        if eigenvalues[0] != eigenvalues[1] != eigenvalues[2]:
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except:
                P_inv = "P^-1"
            if not isinstance(P_inv, str):
                diag = np.einsum("ij,jk,kl->il", eigenvectors, A, P_inv)
                r = []
                for i in range(len(diag)):
                    aux = []
                    for j in range(len(diag)):
                        if i == j:
                            aux.append(f"e^{{{diag[i][j]}t}}")
                        else:
                            aux.append("0")
                    r.append(aux)
                solutions.append(f"x(t) = {eigenvectors} {r} {P_inv} c")
                return solutions
            else:
                solutions.append(
                    f"{eigenvectors}{A}{P_inv}c, ocurrió un error al calcular la inversa y no se puedo seguir el procedimieto"
                )
                return solutions
        if len(count_eigenvalues) == 1 or len(count_eigenvalues) == 2:
            # multiplicad 3 o 2
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except:
                P_inv = "P^-1"

            diag_l = np.zeros((3, 3))
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
                            aux.append(f"e^{{{diag_l[i][j]}t}}")
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
                s = f"x(t) = {eigenvectors} {diag_end} {P_inv} (I + {suma})c"
                solutions.append(s)
                return solutions
            else:
                solutions.append(
                    f"{eigenvectors}{diag_l}{P_inv}c, ocurrió un error al calcular la inversa y no se puedo seguir el procedimieto"
                )
                return solutions

    def system_3x3(self, t, Y, A):
        return A @ Y

    def plot_phase_diagram_3d(self, A, sol_y) -> None:
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
