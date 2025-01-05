import streamlit as st
import numpy as np
import os
import pandas as pd
import altair as alt
from logic.logic import RungeKutta  # Importa RungeKutta
from logic.error import (
    Parentesis_Error,
    RK_Error,
    Inf,
    SEL,
)  # Importa las excepciones personalizadas

st.subheader("Graficadora")


input_col, plot_col = st.columns([1, 2])

with input_col:
    equation_str = st.text_area(
        "Ingresa la ecuación diferencial (ej. 'dy/dx = -2*x*y'):", ""
    )

    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("Valor inicial de x (x0):", value=0.0)
    with col2:
        y0 = st.number_input("Valor inicial de y (y0):", value=1.0)

    h = st.number_input("Paso de integración (h):", value=0.1)
    xf = st.number_input("Valor final de x (xf):", value=10.0)

    if st.button("Resolver"):
        if equation_str:
            try:
                rk_solver = RungeKutta(x0, y0, xf, h, equation_str)
                print(rk_solver.ast)
                print(rk_solver.tokens)
                X, Y = rk_solver.solver()
                x_min, x_max = x0, xf
                y_min, y_max = min(Y), max(Y)
                X_iso, Y_iso, U_iso, V_iso = rk_solver.isoclinas(
                    x_min, x_max, y_min, y_max
                )
                with plot_col:

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

                    arrow_length = 0.1
                    quiver_data["x2"] = (
                        quiver_data["x"] + quiver_data["u"] * arrow_length
                    )
                    quiver_data["y2"] = (
                        quiver_data["y"] + quiver_data["v"] * arrow_length
                    )

                    arrows = (
                        alt.Chart(quiver_data)
                        .mark_line(color="red")
                        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
                    )

                    # Combinar ambos gráficos
                    combined_chart = alt.layer(line_chart, arrows).properties(
                        width=600, height=400
                    )
                    st.altair_chart(combined_chart, use_container_width=True)

            except ValueError as e:
                st.error(f"Error: {e}")
            except Parentesis_Error as e:
                st.error("Error de paréntesis en la ecuación.")
            except RK_Error as e:
                st.error("Error en el método de Runge-Kutta.")
            except Inf as e:
                st.error("Error de solución infinita.")
            except Exception as e:
                st.error(f"Error inesperado: {e}")
        else:
            st.write("Por favor, ingresa una ecuación.")
