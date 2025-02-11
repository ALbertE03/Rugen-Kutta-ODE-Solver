import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from logic.logic import RungeKutta
from logic.error import (
    Parentesis_Error,
    RK_Error,
    Inf,
)
import matplotlib.pyplot as plt
import sympy as sp

st.subheader("Graficadora")

input_col, plot_col = st.columns([1, 2])

with input_col:

    equation_str = st.text_input(
        "Ingresa la ecuación diferencial (ej. 'sen(x-y)'):", ""
    )

    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("Valor inicial de x (x0):", value=0.0)
    with col2:
        y0 = st.number_input("Valor inicial de y (y0):", value=1.0)

    h = st.number_input("Paso de integración (h):", value=0.1)
    xf = st.number_input("Valor final de x:", value=10.0)
    isoclinas = st.toggle("isoclinas")
    col1, col2 = st.columns(2)
    resolver_pressed = col1.button("Resolver RK-4")
    comparar_pressed = col2.button("Comparar RK-3")
with st.expander("ℹ️ **Información de Entradas Válidas**"):
    st.markdown(
        """
        <h2 style='text-align: center;'>✏️ <b>Entradas Válidas</b></h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h3 style='text-align: center;'>📌 <b>Términos</b></h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        <div style="text-align: center; font-size: 16px;">
            Solo se podrá escribir en términos de <code>x</code> e <code>y</code> en cualquier combinación en las funciones y los operadores.  
            También puede usar los números π usando la palabra <code>pi</code> y el número <code>e</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h3 style='text-align: center;'>📐 <b>Funciones</b></h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h4 style='text-align: center;'>🔹 <b>Trigonométricas</b></h4>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        \begin{aligned}
        \text{sen}(x) &\quad \text{Representa el seno.} \\
        \text{cos}(x) &\quad \text{Representa el coseno.} \\
        \text{tan}(x) &\quad \text{Representa la tangente.} \\
        \text{cot}(x) &\quad \text{Representa la cotangente.} \\
        \end{aligned}
        """
    )

    st.markdown(
        """
        <h4 style='text-align: center;'>🔹 <b>Inversas</b></h4>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        \begin{aligned}
        \arctan(x) &\quad \text{Representa la arcotangente.} \\
        \arcsin(x) &\quad \text{Representa el arcseno.} \\
        \arccos(x) &\quad \text{Representa el arccoseno.} \\
        \end{aligned}
        """
    )

    st.markdown(
        """
        <h4 style='text-align: center;'>🔹 <b>Logarítmicas</b></h4>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        \begin{aligned}
        \ln(x) &\quad \text{Representa el logaritmo natural.} \\
        \log(x) &\quad \text{Representa el logaritmo en base 10.} \\
        \end{aligned}
        """
    )

    st.markdown(
        """
        <h4 style='text-align: center;'>🔹 <b>Exponenciales</b></h4>
        """,
        unsafe_allow_html=True,
    )

    st.latex(r"f(x)^{g(x)}")

    st.markdown(
        """
        <h3 style='text-align: center;'>➕ <b>Operadores</b></h3>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        \begin{array}{c}
        \text{\textbf{Operadores Válidos}} \\[8pt]
        \begin{aligned}
        + & : \text{ \textbf{Representa la suma.} } \\[4pt]
        - & : \text{ \textbf{Representa la resta.} } \\[4pt]
        * & : \text{ \textbf{Representa la multiplicación.}} \\[4pt]
        / & : \text{ \textbf{Representa la división.} } \\[4pt]
        \wedge \, \text{o} \, ** & : \text{ \textbf{Representan la potenciación.} }
        \end{aligned}
        \end{array}
        """
    )

    st.markdown(
        """
    <h4 style='text-align: center;'>🔍 <b>Detalles sobre la Multiplicación:</b></h4>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: center;">

         **`#(...)`** y **`(...)#`**: *Será tomado en cuenta como* `#*(...)` *y* `(...)*#`

         **`#x`**: *Será tomado en cuenta como* `#*x` *y* `x*#`

         **`x(...)`** y **`(...)x`**: *Será tomado en cuenta como* `x*(...)` *y* `(...)*x`

         **EN CUALQUIER OTRO CASO, DEBERÁ INDICAR LA MULTIPLICACIÓN EXPLÍCITAMENTE.**
        </div>
        """,
        unsafe_allow_html=True,
    )


with plot_col:
    if resolver_pressed or comparar_pressed:
        if equation_str:
            try:
                rk_solver = RungeKutta(x0, y0, xf, h, equation_str)
                print(f"f: parseada: {rk_solver.ast}")
                X, Y = rk_solver.solver_rk4()
                x_min, x_max = x0, xf
                y_min, y_max = min(Y), max(Y)

                scale_factor = max(x_max - x_min, y_max - y_min) / 1
                X_iso, Y_iso, U_iso, V_iso = rk_solver.isoclinas(
                    x_min, x_max, y_min, y_max
                )

                line_data = pd.DataFrame({"x": X, "y": Y})

                quiver_data = pd.DataFrame(
                    {
                        "x": X_iso,
                        "y": Y_iso,
                        "u": U_iso,
                        "v": V_iso,
                    }
                )

                arrow_length = 0.2 * scale_factor
                quiver_data["x2"] = quiver_data["x"] + quiver_data["u"] * arrow_length
                quiver_data["y2"] = quiver_data["y"] + quiver_data["v"] * arrow_length

                quiver_data["x2"] = quiver_data["x2"].clip(lower=x_min, upper=x_max)
                quiver_data["y2"] = quiver_data["y2"].clip(lower=y_min, upper=y_max)

                fig, ax = plt.subplots()

                ax.set_facecolor("#0E1117")
                fig.patch.set_facecolor("#0E1117")
                # Graficar los datos
                ax.plot(X, Y, label="rk-4", color="blue")
                ax.spines["top"].set_color("white")
                ax.spines["right"].set_color("white")
                ax.spines["bottom"].set_color("white")
                ax.spines["left"].set_color("white")

                # Cambiar el color de las marcas y etiquetas de los ejes
                ax.tick_params(
                    axis="both", colors="white"
                )  # Cambiar color de los valores en los ejes

                # Cambiar el color de las etiquetas de los ejes a blanco
                ax.set_xlabel("X-axis", color="white")
                ax.set_ylabel("Y-axis", color="white")
                scale = 1
                if xf > 50:
                    scale = 0.5
                if xf > 80:
                    scale = 0.1
                if isoclinas:
                    plt.quiver(
                        quiver_data["x"],
                        quiver_data["y"],
                        quiver_data["u"],
                        quiver_data["v"],
                        angles="xy",
                        scale_units="xy",
                        scale=scale,
                        color="red",
                    )

                if comparar_pressed:
                    rk_solver_3 = RungeKutta(x0, y0, xf, h, equation_str)
                    X3, Y3 = rk_solver_3.solver_rk3()

                    y_min = min(y_min, min(Y3))
                    y_max = max(y_max, max(Y3))

                    line_data3 = pd.DataFrame({"x": X3, "y": Y3})
                    plt.plot(
                        line_data3["x"],
                        line_data3["y"],
                        label="rk-3",
                        color="green",
                    )

                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)

                plt.xlabel("X")
                plt.ylabel("Y")
                plt.legend()

                st.pyplot(plt)

            except ValueError as e:
                st.error(f"Error: {e}")
            except Parentesis_Error:
                st.error("Error de paréntesis en la ecuación.")
            except RK_Error:
                st.error("Error en el método de Runge-Kutta.")
            except Inf:
                st.error("Error de solución infinita.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Por favor, ingresa una ecuación.")
