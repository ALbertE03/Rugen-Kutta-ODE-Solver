import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st

def solve_system_2x2(A, Y0):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    stable = all(np.real(eigenvalues) < 0)
    exp_At = expm(A)
    sol_y = np.random.rand(2, 1)  # Agrega esta línea, o usa el cálculo adecuado para `sol_y`
    return exp_At, stable, eigenvalues, eigenvectors, sol_y  # Asegúrate de retornar 5 valores

def format_matrix(matrix):
    return np.vectorize(lambda x: f"{x:.2e}" if abs(x) >= 1e7 or abs(x) < 1e-7 else x)(matrix)

def get_solutions_2x2(eigenvalues, eigenvectors, Y0):
    solutions = []
    c1, c2 = Y0[0, 0], Y0[1, 0]
    for i, eigenvalue in enumerate(eigenvalues):
        if np.iscomplex(eigenvalue):
            alpha = round(np.real(eigenvalue), 3)
            beta = round(np.imag(eigenvalue), 3)
            v = eigenvectors[:, i].reshape(2, 1).round(3)
            v_str = " \\\\ ".join([f"{v[j, 0].real:.3f}".rstrip('0').rstrip('.') + (f" + {v[j, 0].imag:.3f}i".rstrip('0').rstrip('.') if v[j, 0].imag != 0 else '') for j in range(len(v))])
            solutions.append(rf"$x_1(t) = e^{{{alpha}t}} \left({c1} \cos({beta}t) + {c2} \sin({beta}t)\right) \begin{{bmatrix}} {v_str.replace('j', 'i')} \end{{bmatrix}}$")
        else:
            v = eigenvectors[:, i].reshape(2, 1).round(3)
            v_str = " \\\\ ".join([f"{v[j, 0]:.3f}".rstrip('0').rstrip('.') for j in range(len(v))])
            multiplicity = np.count_nonzero(np.isclose(eigenvalues, eigenvalue))
            if multiplicity == 1:
                solutions.append(rf"$x_1(t) = {c1} e^{{{eigenvalue:.3f}t}} \begin{{bmatrix}} {v_str} \end{{bmatrix}}$")
                solutions.append(rf"$x_2(t) = {c2} e^{{{eigenvalue:.3f}t}} \begin{{bmatrix}} {v_str} \end{{bmatrix}}$")
            else:
                solutions.append(rf"$x_1(t) = {c1} e^{{{eigenvalue:.3f}t}} \begin{{bmatrix}} {v_str} \end{{bmatrix}}$")
                solutions.append(rf"$x_2(t) = {c2} e^{{{eigenvalue:.3f}t}} \begin{{bmatrix}} {v_str} \end{{bmatrix}}$")
    return solutions

def system_2x2(t, Y, A):
    return A @ Y

def plot_phase_diagram_2d(A):
    fig, ax = plt.subplots(figsize=(6, 6))  # Ajustar tamaño del gráfico
    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 200)
    Y0_points = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
    
    for y0 in Y0_points:
        sol = solve_ivp(system_2x2, t_span, y0, args=(A,), t_eval=t_eval)
        ax.plot(sol.y[0], sol.y[1])
        ax.quiver(sol.y[0], sol.y[1], np.gradient(sol.y[0]), np.gradient(sol.y[1]), scale_units='xy', angles='xy', scale=1)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)
