import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st

def solve_system_2x2(A, Y0):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    stable = all(np.real(eigenvalues) < 0)
    exp_At = expm(A)
    sol_y = np.random.rand(2)  # Agrega esta línea, o usa el cálculo adecuado para `sol_y`
    return exp_At, stable, eigenvalues, eigenvectors, sol_y  # Asegúrate de retornar 5 valores


def get_solutions_2x2(eigenvalues, eigenvectors, Y0):
    solutions = []
    for i, eigenvalue in enumerate(eigenvalues):
        if np.iscomplex(eigenvalue):
            alpha = np.real(eigenvalue)
            beta = np.imag(eigenvalue)
            v = eigenvectors[:, i]
            v_str = " \\\\ ".join([f"{v[j].real:.2f} + {v[j].imag:.2f}i" for j in range(len(v))])
            solutions.append(rf"f_{{{i+1}}}(t) = e^{{{alpha}t}} \left(c_{{{i*2+1}}} \cos({beta}t) + c_{{{i*2+2}}} \sin({beta}t)\right) \begin{{bmatrix}} {v_str} \end{{bmatrix}}")
        else:
            v = eigenvectors[:, i]
            v_str = " \\\\ ".join([f"{v[j]:.2f}" for j in range(len(v))])
            multiplicity = np.count_nonzero(np.isclose(eigenvalues, eigenvalue))
            if multiplicity == 1:
                solutions.append(rf"f_{{{i+1}}}(t) = c_{{{i+1}}} e^{{{eigenvalue}t}} \begin{{bmatrix}} {v_str} \end{{bmatrix}}")
            else:
                for j in range(multiplicity):
                    solutions.append(rf"f_{{{i+1+j}}}(t) = \left(c_{{{i+1+j}}} + c_{{{i+1+j+1}}} t\right) e^{{{eigenvalue}t}} \begin{{bmatrix}} {v_str} \end{{bmatrix}}")
    return solutions

def system_2x2(t, Y, A):
    return A @ Y

def plot_phase_diagram_2d(A):
    fig, ax = plt.subplots(figsize=(4, 3))
    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 200)
    Y0 = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
    
    for y0 in Y0:
        sol = solve_ivp(system_2x2, t_span, y0, args=(A,), t_eval=t_eval)
        ax.plot(sol.y[0], sol.y[1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Diagrama de Fase en 2D")
    st.pyplot(fig)
