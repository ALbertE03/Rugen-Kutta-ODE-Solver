import streamlit as st
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Función para resolver el sistema de ecuaciones diferenciales
def solve_system(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    stable = all(np.real(eigenvalues) < 0)
    exp_At = expm(A)
    return exp_At, stable, eigenvalues, eigenvectors

# Función para determinar las soluciones del sistema
def get_solutions(eigenvalues, eigenvectors):
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

# Función para el sistema de ecuaciones diferenciales
def system(t, Y, A):
    return A @ Y

# Función para visualizar el diagrama de fase en 2D
def plot_phase_diagram_2d(A):
    fig, ax = plt.subplots(figsize=(4, 3))
    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 200)
    Y0 = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
    
    for y0 in Y0:
        sol = solve_ivp(system, t_span, y0, args=(A,), t_eval=t_eval)
        ax.plot(sol.y[0], sol.y[1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Diagrama de Fase en 2D")
    st.pyplot(fig)

# Función para visualizar el diagrama de fase en 3D
def plot_phase_diagram_3d(A):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')
    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 200)
    Y0 = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    
    for y0 in Y0:
        sol = solve_ivp(system, t_span, y0, args=(A,), t_eval=t_eval)
        ax.plot(sol.y[0], sol.y[1], sol.y[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Diagrama de Fase en 3D")
    st.pyplot(fig)

# Configura la página
st.title("Solver de Ecuaciones Diferenciales")

# Selecciona el tamaño de la matriz
option = st.radio("Seleccione el tamaño del sistema:", ("2x2", "3x3"))

col1, col2 = st.columns(2)

if option == "2x2":
    col1.subheader("Entradas de la matriz 2x2")
    matrix = []
    row1 = col1.columns(2)
    row2 = col1.columns(2)
    matrix.append([row1[0].number_input("A[1,1]", format="%.2f", key="A11"),
                   row1[1].number_input("A[1,2]", format="%.2f", key="A12")])
    matrix.append([row2[0].number_input("A[2,1]", format="%.2f", key="A21"),
                   row2[1].number_input("A[2,2]", format="%.2f", key="A22")])
    A = np.array(matrix)
    if col1.button("Graficar"):
        col1.subheader("Diagrama de Fase 2D")
        plot_phase_diagram_2d(A)
    
    col2.subheader("Resultados")
    res1, res2 = col2.columns(2)
    res3, res4 = col2.columns(2)
    exp_At, stable, eigenvalues, eigenvectors = solve_system(A)
    res1.write("Matriz exponencial \(e^{At}\):")
    res1.write(exp_At)
    res2.write("Estabilidad:")
    res2.write("El sistema es estable" if stable else "El sistema no es estable")
    if not stable:
        res2.write("El sistema no es estable porque al menos uno de los valores propios tiene parte real no negativa.")
    res3.write("Valores propios:")
    res3.write(eigenvalues)
    res4.write("Vectores propios:")
    res4.write(eigenvectors)
    res3.write("Soluciones del sistema:")
    solutions = get_solutions(eigenvalues, eigenvectors)
    for solution in solutions:
        res3.latex(solution)

elif option == "3x3":
    col1.subheader("Entradas de la matriz 3x3")
    matrix = []
    row1 = col1.columns(3)
    row2 = col1.columns(3)
    row3 = col1.columns(3)
    matrix.append([row1[0].number_input("A[1,1]", format="%.2f", key="A31"),
                   row1[1].number_input("A[1,2]", format="%.2f", key="A32"),
                   row1[2].number_input("A[1,3]", format="%.2f", key="A33")])
    matrix.append([row2[0].number_input("A[2,1]", format="%.2f", key="A41"),
                   row2[1].number_input("A[2,2]", format="%.2f", key="A42"),
                   row2[2].number_input("A[2,3]", format="%.2f", key="A43")])
    matrix.append([row3[0].number_input("A[3,1]", format="%.2f", key="A51"),
                   row3[1].number_input("A[3,2]", format="%.2f", key="A52"),
                   row3[2].number_input("A[3,3]", format="%.2f", key="A53")])
    A = np.array(matrix)
    if col1.button("Graficar"):
        col1.subheader("Diagrama de Fase 3D")
        plot_phase_diagram_3d(A)
    
    col2.subheader("Resultados")
    res1, res2 = col2.columns(2)
    res3, res4 = col2.columns(2)
    exp_At, stable, eigenvalues, eigenvectors = solve_system(A)
    res1.write("Matriz exponencial \(e^{At}\):")
    res1.write(exp_At)
    res2.write("Estabilidad:")
    res2.write("El sistema es estable" if stable else "El sistema no es estable")
    if not stable:
        res2.write("El sistema no es estable porque al menos uno de los valores propios tiene parte real no negativa.")
    res3.write("Valores propios:")
    res3.write(eigenvalues)
    res4.write("Vectores propios:")
    res4.write(eigenvectors)
    res3.write("Soluciones del sistema:")
    solutions = get_solutions(eigenvalues, eigenvectors)
    for solution in solutions:
        res3.latex(solution)
