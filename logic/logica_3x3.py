import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import streamlit as st

def matrix_exponential(A, t):
    n = A.shape[0]
    exp_At = np.eye(n) + A * t
    for k in range(2, 20):  # Ajustar el número de términos para mayor precisión
        exp_At += np.linalg.matrix_power(A * t, k) / np.math.factorial(k)
    return exp_At

def solve_system_3x3(A, Y0):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    stable = all(np.real(eigenvalues) < 0)
    exp_At = matrix_exponential(A, 1)  # Calcular la matriz exponencial para t=1
    
    # Resolver el sistema usando los valores iniciales
    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 200)
    sol = solve_ivp(system_3x3, t_span, Y0, args=(A,), t_eval=t_eval)
    
    return exp_At, stable, eigenvalues, eigenvectors, sol.y

def get_solutions_3x3(eigenvalues, eigenvectors, Y0):
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

def system_3x3(t, Y, A):
    return A @ Y

def plot_phase_diagram_3d(A, sol_y):
    fig = go.Figure()

    # Crear líneas y puntos coloridos para la solución particular
    fig.add_trace(go.Scatter3d(x=sol_y[0], y=sol_y[1], z=sol_y[2],
                               mode='lines+markers',
                               marker=dict(size=4),
                               line=dict(color='blue', width=2)))

    fig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'),
                      title='Diagrama de Fase en 3D')
    st.plotly_chart(fig)
