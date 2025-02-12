import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from logic.logic_2x2 import Solve2x2
from logic.logic_3x3 import Solve3x3
import sympy as sp

solve2x2 = Solve2x2()
solve3x3 = Solve3x3()

st.title("Sistemas de Ecuaciones Diferenciales")
option = st.radio("Seleccione el tamaño del sistema:", ("2x2", "3x3"))
col1, col2 = st.columns(2)

if option == "2x2":
    try:
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
        tr = False
        for i in A:
            for j in i:
                if j != 0:
                    tr = True
                    break
            if tr:
                break
        if col1.button("Graficar"):
            if tr:
                col1.markdown("### Diagrama de Fase 2D")
                with col1:
                    solve2x2.plot_phase_diagram_2d(A)

                col2.markdown("### Resultados")
                res1, res2 = col2.columns(2)
                exp_At, stable, eigenvalues, eigenvectors, sol_y = (
                    solve2x2.solve_system_2x2(A)
                )

                formatted_matrix = np.vectorize(
                    lambda x: (
                        f"{x:.2e}" if abs(x) >= 1e7 or abs(x) < 1e-7 else f"{x:.2f}"
                    )
                )(exp_At)
                latex_matrix = (
                    r"\begin{bmatrix}"
                    + r" \\ ".join(
                        [" & ".join(map(str, row)) for row in formatted_matrix]
                    )
                    + r"\end{bmatrix}"
                )
                res1.markdown("##### Matriz exponencial  $e^{At}$:")
                res1.latex(latex_matrix)
                res2.markdown("#### Estabilidad:")
                res2.write(
                    "El sistema es estable" if stable else "El sistema no es estable"
                )
                l1, l2 = eigenvalues
                if np.isreal(l1) and np.isreal(l2):
                    l_r = np.real(l1)
                    l2_r = np.real(l2)
                    if (l_r <= 0 and l2_r <= 0) or (l_r >= 0 and l2_r >= 0):
                        if l_r == l2_r:
                            res2.write("Clasificación: Nodo propio")
                        else:
                            res2.write("Clasificación: Nodo impropio")
                    elif (l_r < 0 and l2_r > 0) or (l_r > 0 and l2_r < 0):
                        res2.write("Punto de silla")
                elif np.iscomplex(l1) and np.iscomplex(l2):
                    a = np.real(l1)
                    if np.isclose(a, 0):
                        res2.write("Clasificación: Centro")
                    else:
                        res2.write("Clasificación: Foco")

                col2.markdown("---")
                res3, res4 = col2.columns(2)
                res3.markdown("#### Valores propios:")
                eigenvalues_matrix = (
                    r"\begin{bmatrix}"
                    + r" \\ ".join(
                        [
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
                    )
                    + r"\end{bmatrix}"
                )
                res3.latex(eigenvalues_matrix)

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
                eigenvectors_matrix = (
                    r"\begin{bmatrix}"
                    + r" \\ ".join([" & ".join(row) for row in eigenvectors_display])
                    + r"\end{bmatrix}"
                )

                res4.latex(eigenvectors_matrix)
                col2.markdown("----")
                col2.markdown("#### Soluciones del sistema:")
                solutions = solve2x2.get_solutions_2x2(eigenvalues, eigenvectors, A)
                col2.markdown(f"${sp.latex(solutions[0])}$")
                col2.markdown(f"${sp.latex(solutions[1])}$")
            else:
                col2.warning("introduzca valores a la matriz")
    except:
        col2.warning("Matriz con valores muy altos")
elif option == "3x3":
    try:
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
        tr = False
        for i in A:
            for j in i:
                if j != 0:
                    tr = True
                    break

            if tr:
                break
        if col1.button("Graficar"):
            if tr:
                Y0 = np.random.rand(3)
                exp_At, stable, eigenvalues, eigenvectors, sol_y = (
                    solve3x3.solve_system_3x3(A, Y0)
                )

                col1.markdown("### Diagrama de Fase 3D")
                with col1:
                    solve3x3.plot_phase_diagram_3d(A, sol_y)

                col2.markdown("### Resultados")
                res1, res2 = col2.columns(2)
                res1.markdown("##### Matriz exponencial  $e^{At}$:")
                formatted_matrix = np.vectorize(
                    lambda x: (
                        f"{x:.2e}" if abs(x) >= 1e7 or abs(x) < 1e-7 else f"{x:.2f}"
                    )
                )(exp_At)
                latex_matrix = (
                    r"\begin{bmatrix}"
                    + r" \\ ".join(
                        [" & ".join(map(str, row)) for row in formatted_matrix]
                    )
                    + r"\end{bmatrix}"
                )
                res1.latex(latex_matrix)
                res2.markdown("#### Estabilidad:")
                res2.write(
                    "El sistema es estable" if stable else "El sistema no es estable"
                )

                l1, l2, l3 = eigenvalues
                real = np.isreal(eigenvalues)
                if all(real):
                    l1_r = np.real(l1)
                    l2_r = np.real(l2)
                    l3_r = np.real(l3)
                    if l1_r < 0 and l2_r < 0 and l3_r < 0:
                        res2.write("Clasificación: Nodo")
                    else:
                        res2.write("Clasificación: Punto de Silla")
                if sum(real) == 1:
                    real_l = np.real(eigenvalues[real][0])
                    complex_l = eigenvalues[~real]
                    a = np.real(complex_l[0])
                    if np.isclose(a, 0):
                        res2.write("Clasificación: Centro")
                    else:
                        res2.write("Clasificación: Foco")

                col2.markdown("---")
                res3, res4 = col2.columns(2)
                res3.markdown("#### Valores propios:")
                eigenvalues_matrix = (
                    r"\begin{bmatrix}"
                    + r" \\ ".join(
                        [
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
                    )
                    + r"\end{bmatrix}"
                )
                res3.latex(eigenvalues_matrix)

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
                eigenvectors_matrix = (
                    r"\begin{bmatrix}"
                    + r" \\ ".join([" & ".join(row) for row in eigenvectors_display])
                    + r"\end{bmatrix}"
                )
                res4.latex(eigenvectors_matrix)
                col2.markdown("---")
                col2.markdown("#### Soluciones del sistema:")

                solutions = solve3x3.get_solutions_3x3(eigenvalues, eigenvectors, A)
                col2.markdown(f"${sp.latex(solutions[0])}$")
                col2.markdown(f"${sp.latex(solutions[1])}$")
                col2.markdown(f"${sp.latex(solutions[2])}$")

            else:
                col2.warning("introduzca valores a la matriz")
    except:
        col2.warning("Matriz con valores muy altos")
