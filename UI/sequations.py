import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from logic.logic_2x2 import Solve2x2
from logic.logic_3x3 import Solve3x3

solve2x2 = Solve2x2()
solve3x3 = Solve3x3()

st.title("Sistemas de Ecuaciones Diferenciales")
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
        res1.markdown("##### Matriz exponencial  $e^{At}$:")
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
            (
                f"{val.real:.3f}".rstrip("0").rstrip(".")
                + (
                    f" + {val.imag:.3f}i".rstrip("0").rstrip(".")
                    if val.imag > 0
                    else f" - {-val.imag:.3f}i".rstrip("0").rstrip(".")
                )
                if val.imag != 0
                else f"{val.real:.3f}".rstrip("0").rstrip(".")
            )
            for val in eigenvalues
        ]
        res3.table(eigenvalues_display)

        res4.markdown("#### Vectores propios:")
        eigenvectors_display = [
            [
                (
                    f"{val.real:.3f} + {val.imag:.3f}i".rstrip("0").rstrip(".")
                    if val.imag != 0
                    else f"{val.real:.3f}".rstrip("0").rstrip(".")
                )
                for val in row
            ]
            for row in eigenvectors
        ]
        res4.table(eigenvectors_display)

        res3.markdown("#### Soluciones del sistema:")
        solutions = solve2x2.get_solutions_2x2(eigenvalues, eigenvectors, A)
        for solution in solutions:
            
            col2.markdown(f"${solution}$")


elif option == "3x3":
    col1.markdown("### Entradas de la matriz 3x3")
    matrix = []
    row1 = col1.columns(3)
    row2 = col1.columns(3)
    row3 = col1.columns(3)

    matrix.append(
        [
            row1[0].number_input("A[1,1]", format="%.2f", key="A11_3x3"),
            row1[1].number_input("A[1,2]", format="%.2f", key="A12_3x3"),
            row1[2].number_input("A[1,3]", format="%.2f", key="A13_3x3"),
        ]
    )
    matrix.append(
        [
            row2[0].number_input("A[2,1]", format="%.2f", key="A21_3x3"),
            row2[1].number_input("A[2,2]", format="%.2f", key="A22_3x3"),
            row2[2].number_input("A[2,3]", format="%.2f", key="A23_3x3"),
        ]
    )
    matrix.append(
        [
            row3[0].number_input("A[3,1]", format="%.2f", key="A31_3x3"),
            row3[1].number_input("A[3,2]", format="%.2f", key="A32_3x3"),
            row3[2].number_input("A[3,3]", format="%.2f", key="A33_3x3"),
        ]
    )
    A = np.array(matrix)

    if col1.button("Graficar"):
       
        Y0 = np.random.rand(3)
        exp_At, stable, eigenvalues, eigenvectors, sol_y = solve3x3.solve_system_3x3(A, Y0)

        col1.markdown("### Diagrama de Fase 3D")
        with col1:
            solve3x3.plot_phase_diagram_3d(A, sol_y)

        col2.markdown("### Resultados")
        res1, res2 = col2.columns(2)
        res1.markdown("##### Matriz exponencial  $e^{At}$:")
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
            (
                f"{val.real:.3f}".rstrip("0").rstrip(".")
                + (
                    f" + {val.imag:.3f}i".rstrip("0").rstrip(".")
                    if val.imag > 0
                    else f" - {-val.imag:.3f}i".rstrip("0").rstrip(".")
                )
                if val.imag != 0
                else f"{val.real:.3f}".rstrip("0").rstrip(".")
            )
            for val in eigenvalues
        ]
        res3.table(eigenvalues_display)

        res4.markdown("#### Vectores propios:")
        eigenvectors_display = [
            [
                (
                    f"{val.real:.3f} + {val.imag:.3f}i".rstrip("0").rstrip(".")
                    if val.imag != 0
                    else f"{val.real:.3f}".rstrip("0").rstrip(".")
                )
                for val in row
            ]
            for row in eigenvectors
        ]
        res4.table(eigenvectors_display)

        res3.markdown("#### Soluciones del sistema:")
        
        solutions = solve3x3.get_solutions_3x3(eigenvalues, eigenvectors, Y0)
        for solution in solutions:
            
            col2.markdown(solution, unsafe_allow_html=True)
