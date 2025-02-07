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
    with st.expander("Información"):
        st.write("jola")

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
                    {"x": X_iso, "y": Y_iso, "u": U_iso, "v": V_iso}
                )

                line_chart = (
                    alt.Chart(line_data)
                    .mark_line()
                    .encode(
                        x=alt.X("x:Q", scale=alt.Scale(domain=(x_min, x_max))),
                        y=alt.Y("y:Q", scale=alt.Scale(domain=(y_min, y_max))),
                    )
                )

                arrow_length = 0.2 * scale_factor
                quiver_data["x2"] = quiver_data["x"] + quiver_data["u"] * arrow_length
                quiver_data["y2"] = quiver_data["y"] + quiver_data["v"] * arrow_length
                quiver_data["x2"] = quiver_data["x2"].clip(lower=x_min, upper=x_max)
                quiver_data["y2"] = quiver_data["y2"].clip(lower=y_min, upper=y_max)

                arrows = (
                    alt.Chart(quiver_data)
                    .mark_line(color="red")
                    .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
                )
                if isoclinas:
                    combined_chart = alt.layer(line_chart, arrows).properties(
                        width=600, height=400
                    )
                else:
                    combined_chart = alt.layer(line_chart).properties(
                        width=600, height=400
                    )

                if comparar_pressed:
                    rk_solver_3 = RungeKutta(x0, y0, xf, h, equation_str)
                    X3, Y3 = rk_solver_3.solver_rk3()

                    y_min = min(y_min, min(Y3))
                    y_max = max(y_max, max(Y3))

                    line_data3 = pd.DataFrame({"x": X3, "y": Y3})
                    line_chart3 = (
                        alt.Chart(line_data3)
                        .mark_line(color="green")
                        .encode(
                            x=alt.X("x:Q", scale=alt.Scale(domain=(x_min, x_max))),
                            y=alt.Y("y:Q", scale=alt.Scale(domain=(y_min, y_max))),
                        )
                    )

                    if isoclinas:
                        combined_chart = alt.layer(line_chart3, arrows).properties(
                            width=600, height=400
                        )
                    else:
                        combined_chart = alt.layer(line_chart3).properties(
                            width=600, height=400
                        )
                st.altair_chart(combined_chart, use_container_width=True)

            except ValueError as e:
                st.error(f"Error: {e}")
            except Parentesis_Error:
                st.error("Error de paréntesis en la ecuación.")
            except RK_Error:
                st.error("Error en el método de Runge-Kutta.")
            except Inf:
                st.error("Error de solución infinita.")
            except Exception as e:
                st.error(f"Error inesperado: {e}")
        else:
            st.info("Por favor, ingresa una ecuación.")
