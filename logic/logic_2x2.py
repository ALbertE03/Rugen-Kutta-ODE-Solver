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
            lambda x: f"{x:.2e}" if abs(x) >= 1e7 or abs(x) < 1e-7 else x
        )(matrix)

    def get_solutions_2x2(self, eigenvalues, eigenvectors, A):
        solutions = []
        if A[0][1] == 0 and A[1][1] == 0:
            """
            es una matriz diagonal y la solución viene dada por
            x(t) = diag[e^l_i]c con l_i los valores de la diagonal
            """
            r = []
            for i in range(len(new_A)):
                aux = []
                for j in range(len(new_A)):
                    if i == j:
                        aux.append(f"e^{new_A[i][j]}t")
                    else:
                        aux.append("0")
                r.append(aux)
            solutions.append(f"x(t)={r}*c")
            return solutions
        if A[0][0] == A[1][1] and A[0][1] + A[1][0] == 0:
            """si es de la forma
            [[a, -b],
            [b, a]] => la solución es e^at *([[cos(bt),sen(bt)], * c
                                            [sen(bt),cos(bt)]])

            que vedría siendo
            x1(t) = e^at*(c1*cos(bt)-c2sen(bt))
            x2(t) = e^at*(c1*sen(bt)+c2cos(bt))
            """
            solutions.append(rf"e^{A[0][0]}t(c_1*cos({A[1][0]}t)-c_2*sen({A[1][0]}t))")
            solutions.append(f"e^{A[0][0]}t(c_1*sen({A[1][0]}t)+c_2*cos({A[1][0]}t))")
            return solutions

        elif eigenvalues[0] == eigenvalues[1]:
            """
            si los dos valores propios son iguales, solo puede ocurrir si son reales ('en matrices 2x2')
            => en este caso la solución viene dada por:
            x(t) = P* diag[e^(l_i*t)]*P^{-1}*[I+Nt+N^2t+N^3t.........(N^{k-1}t^{k-1})/(k-1)!]x0

            P es la matriz de los vectores propios (eigenvectors)
            P^{-1} la inversa
            N = A-S
            donde S = P*diag[l_i]*P^{-1}
            N es una matriz nipotente, al elevarla a un cierto k se vovlerá 0 y la suma infinia será hasta ahí.
            """
            try:
                P_inv = np.linalg.inv(eigenvectors)
                ## si trae valores muy pequeños, lansará execpión xq los conisderara 0 y puede llegar el caso que la tome no invertible
                ## es caso de pasar la solución se dará generica con P^-1
            except:
                P_inv = "P^-1"
            if not isinstance(P_inv, str):
                P_inv = np.linalg.inv(eigenvectors)
                diag_l = np.zeros((2, 2))
                np.fill_diagonal(diag_l, eigenvalues)
                S = np.einsum("ij,jk,kl->il", eigenvectors, diag_l, P_inv)
                N = np.subtract(A, S)
                k = 1
                while np.all(N == 0):
                    N = np.dot(N, N)
                    k += 1
                    if k >= eigenvectors.shape[0]:
                        break

                # construimos la solución
                diag_end = []
                for i in range(len(diag_l)):
                    aux = []
                    for j in range(len(diag_l)):
                        if i == j:
                            aux.append(f"e^{diag_l[i][j]}t")
                        else:
                            aux.append("0")
                    diag_end.append(aux)
                suma = ""
                for i in range(1, k + 1):
                    suma += f"+{N}^{i}t"
                s = f"x(t)= {eigenvectors}*{diag_end}*{P_inv}*[I{suma}]c"
                solutions.append(s)
                return solutions
            else:
                ## caso en que no se pueda calcular la inversa.... falta por hacer
                return solutions.append("")
        elif (
            abs(eigenvalues[0].imag) < 1e-20 and abs(eigenvalues[1].imag) < 1e-20
        ):  # asi se comprueba si son reales? chekea esto...
            # ambos son reales pd: tengo que averiguar si si ambos son complejos entra aqui tambien
            """
            si llegamos aqui es ambos son reales y diferentes:
            en este caso la solución  se calcula PAP^-1
            y se crea un nuevo sistema y = P^-1x que es desacoplado y resuelve como el primer caso
            y quedaria
            x(t)= PyP^-1c
            """
            try:
                P_inv = np.linalg.inv(eigenvectors)
            except:
                P_inv = "P^{-1}"
            if not isinstance(P_inv, str):
                new_A = np.einsum("ij,jk,kl->il", eigenvectors, A, P_inv)
                # new A deberia ser diagonal
                r = []
                for i in range(len(new_A)):
                    aux = []
                    for j in range(len(new_A)):
                        if i == j:
                            aux.append(f"e^{new_A[i][j]}t")
                        else:
                            aux.append("0")
                    r.append(aux)

                solutions.append(f"x(t)={eigenvectors}*{r}*{P_inv}*c")
                return solutions
            else:
                return solutions.append("")
        else:

            """aqui entra cuado son complejos ambos, tengo que averiguar una cosa"""
            return solutions.append("no esta echo")

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
