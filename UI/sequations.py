import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from logic.logic_2x2 import Solve2x2

solve2x2 = Solve2x2()

st.title("Solver de Ecuaciones Diferenciales")

option = st.radio("Seleccione el tamaño del sistema:", ("2x2", "3x3"))

col1, col2 = st.columns(2)

if option == "2x2":
    col1.markdown("### Entradas de la matriz 2x2")
    matrix = []
    row1 = col1.columns(2)
    row2 = col1.columns(2)
    matrix.append(
        [
            row1[0].number_input("A[1,1]", format="%.2f", key="A11"),
            row1[1].number_input("A[1,2]", format="%.2f", key="A12"),
        ]
    )
    matrix.append(
        [
            row2[0].number_input("A[2,1]", format="%.2f", key="A21"),
            row2[1].number_input("A[2,2]", format="%.2f", key="A22"),
        ]
    )
    A = np.array(matrix)

    if col1.button("Graficar"):
        col1.markdown("### Diagrama de Fase 2D")
        with col1:
            solve2x2.plot_phase_diagram_2d(A)

        col2.markdown("### Resultados")
        res1, res2 = col2.columns(2)
        res1.markdown("#### Matriz exponencial:  $e^{At}$:")
        exp_At, stable, eigenvalues, eigenvectors, sol_y = solve2x2.solve_system_2x2(A)

        res1.table(
            np.vectorize(lambda x: f"{x:.2e}" if abs(x) >= 1e7 or abs(x) < 1e-7 else x)(
                exp_At
            )
        )

        res2.markdown("#### Estabilidad:")
        res2.write("El sistema es estable" if stable else "El sistema no es estable")
        if not stable:
            res2.write(
                "El sistema no es estable porque al menos uno de los valores propios tiene parte real no negativa."
            )

        col2.markdown("---")
        res3, res4 = col2.columns(2)
        res3.markdown("#### Valores propios:")
        eigenvalues_display = [
            f"{val.real:.3f}".rstrip("0").rstrip(".")
            + (f" + {val.imag:.3f}i".rstrip("0").rstrip(".") if val.imag != 0 else "")
            for val in eigenvalues
        ]
        res3.table(eigenvalues_display)
        res4.markdown("#### Vectores propios:")
        eigenvectors_display = eigenvectors
        res4.table(eigenvectors_display)
        res3.markdown("#### Soluciones del sistema:")
        solutions = solve2x2.get_solutions_2x2(eigenvalues, eigenvectors, A)
        for solution in solutions:
            res3.markdown(f"${solution}$")
