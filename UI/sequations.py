import streamlit as st
import numpy as np
from logic.logica_2x2 import solve_system_2x2, get_solutions_2x2, plot_phase_diagram_2d
from logic.logica_3x3 import solve_system_3x3, get_solutions_3x3, plot_phase_diagram_3d

st.title("Solver de Ecuaciones Diferenciales")

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
    exp_At, stable, eigenvalues, eigenvectors = solve_system_2x2(A)
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
    solutions = get_solutions_2x2(eigenvalues, eigenvectors)
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
    exp_At, stable, eigenvalues, eigenvectors = solve_system_3x3(A)
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
    solutions = get_solutions_3x3(eigenvalues, eigenvectors)
    for solution in solutions:
        res3.latex(solution)
