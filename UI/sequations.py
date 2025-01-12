import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from logic.logica_2x2 import solve_system_2x2, get_solutions_2x2, plot_phase_diagram_2d, system_2x2
from logic.logica_3x3 import solve_system_3x3, get_solutions_3x3, plot_phase_diagram_3d

st.title("Solver de Ecuaciones Diferenciales")

option = st.radio("Seleccione el tamaño del sistema:", ("2x2", "3x3"))

col1, col2 = st.columns(2)

col1, col2 = st.columns(2)

if option == "2x2":
    col1.markdown("### Entradas de la matriz 2x2")
    matrix = []
    row1 = col1.columns(2)
    row2 = col1.columns(2)
    matrix.append([row1[0].number_input("A[1,1]", format="%.2f", key="A11"), row1[1].number_input("A[1,2]", format="%.2f", key="A12")])
    matrix.append([row2[0].number_input("A[2,1]", format="%.2f", key="A21"), row2[1].number_input("A[2,2]", format="%.2f", key="A22")])
    A = np.array(matrix)
    Y0 = [col1.number_input(f"Valor inicial de x{i+1}", format="%.2f") for i in range(2)]
    
    if col1.button("Graficar"):
        col1.markdown("### Diagrama de Fase 2D")
        fig_2d, ax = plt.subplots(figsize=(5, 4))  # Ajustar tamaño del gráfico
        t_span = [0, 10]
        t_eval = np.linspace(0, 10, 200)
        Y0_points = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
        
        for y0 in Y0_points:
            sol = solve_ivp(system_2x2, t_span, y0, args=(A,), t_eval=t_eval)
            ax.plot(sol.y[0], sol.y[1])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title("Diagrama de Fase en 2D")
        col1.pyplot(fig_2d)  # Graficar en la primera columna
        
        col2.markdown("### Resultados")
        res1, res2 = col2.columns(2)
        res1.markdown("#### Matriz exponencial \(e^{At}\):")
        exp_At, stable, eigenvalues, eigenvectors, sol_y = solve_system_2x2(A, Y0)

        res1.table(np.round(exp_At, 3).tolist())  # Mostrar matriz en formato de tabla y truncar a 3 decimales
        res2.markdown("#### Estabilidad:")
        res2.write("El sistema es estable" if stable else "El sistema no es estable")
        if not stable:
            res2.write("El sistema no es estable porque al menos uno de los valores propios tiene parte real no negativa.")
        
        col2.markdown("---")
        res3, res4 = col2.columns(2)
        res3.markdown("#### Valores propios:")
        res3.table(np.round(eigenvalues, 3))  # Mostrar valores propios truncados a 3 decimales
        res4.markdown("#### Vectores propios:")
        res4.table(np.round(eigenvectors, 3))  # Mostrar vectores propios truncados a 3 decimales
        res3.markdown("#### Soluciones del sistema:")
        solutions = get_solutions_2x2(eigenvalues, eigenvectors, Y0)
        for solution in solutions:
            res3.latex(solution)

    def get_solutions_2x2(eigenvalues, eigenvectors, Y0):
        solutions = []
        for i, eigenvalue in enumerate(eigenvalues):
            if np.iscomplex(eigenvalue):
                alpha = np.real(eigenvalue)
                beta = np.imag(eigenvalue)
                v = eigenvectors[:, i]
                v_str = " \\\\ ".join([f"{v[j].real:.2f} + {v[j].imag:.2f}i" for j in range(len(v))])
                solutions.append(rf"f_{{{i+1}}}(t) = e^{{{alpha}t}} \left(c_{{{i*2+1}}} \cos({beta}t) + c_{{{i*2+2}}} \sin({beta}t)\right) \begin{{bmatrix}} {v_str.replace('j', 'i')} \end{{bmatrix}}")
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

elif option == "3x3":
    col1.markdown("### Entradas de la matriz 3x3")
    matrix = []
    row1 = col1.columns(3)
    row2 = col1.columns(3)
    row3 = col1.columns(3)
    matrix.append([row1[0].number_input("A[1,1]", format="%.2f", key="A31"), row1[1].number_input("A[1,2]", format="%.2f", key="A32"), row1[2].number_input("A[1,3]", format="%.2f", key="A33")])
    matrix.append([row2[0].number_input("A[2,1]", format="%.2f", key="A41"), row2[1].number_input("A[2,2]", format="%.2f", key="A42"), row2[2].number_input("A[2,3]", format="%.2f", key="A43")])
    matrix.append([row3[0].number_input("A[3,1]", format="%.2f", key="A51"), row3[1].number_input("A[3,2]", format="%.2f", key="A52"), row3[2].number_input("A[3,3]", format="%.2f", key="A53")])
    A = np.array(matrix)
    Y0 = [col1.number_input(f"Valor inicial de x{i+1}", format="%.2f") for i in range(3)]
    
    if col1.button("Graficar"):

        exp_At, stable, eigenvalues, eigenvectors, sol_y = solve_system_3x3(A, Y0)
    

        st.markdown("#### Soluciones del sistema:")
        solutions = get_solutions_3x3(eigenvalues, eigenvectors, Y0)
        cols = st.columns(3) 

        for i, solution in enumerate(solutions):
            with cols[i % 3]:
                st.latex(solution)


        plot_phase_diagram_3d(A, sol_y)
        
        col2.markdown("### Resultados")
        res1, res2 = col2.columns(2)
        res1.markdown("#### Matriz exponencial \(e^{At}\):")
        res1.table(np.round(exp_At, 3).tolist())  # Mostrar matriz en formato de tabla y truncar a 3 decimales
        res2.markdown("#### Estabilidad:")
        res2.write("El sistema es estable" if stable else "El sistema no es estable")
        if not stable:
            res2.write("El sistema no es estable porque al menos uno de los valores propios tiene parte real no negativa.")
        
        col2.markdown("---")
        res3, res4 = col2.columns(2)
        res3.markdown("#### Valores propios:")
        res3.table(np.round(eigenvalues, 3))  # Mostrar valores propios truncados a 3 decimales
        res4.markdown("#### Vectores propios:")
        res4.table(np.round(eigenvectors, 3))  # Mostrar vectores propios truncados a 3 decimales
        

    def get_solutions_3x3(eigenvalues, eigenvectors, Y0):
        solutions = []
        for i, eigenvalue in enumerate(eigenvalues):
            if np.iscomplex(eigenvalue):
                alpha = np.real(eigenvalue)
                beta = np.imag(eigenvalue)
                v = eigenvectors[:, i]
                v_str = " \\\\ ".join([f"{v[j].real:.2f} + {v[j].imag:.2f}i" for j in range(len(v))])
                solutions.append(rf"f_{{{i+1}}}(t) = e^{{{alpha}t}} \left(c_{{{i*2+1}}} \cos({beta}t) + c_{{{i*2+2}}} \sin({beta}t)\right) \begin{{bmatrix}} {v_str.replace('j', 'i')} \end{{bmatrix}}")
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

