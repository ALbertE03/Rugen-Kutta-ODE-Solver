import streamlit as st
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# import sys
import os

# Agrega la ruta dinámica del directorio 'logic' para importar las clases necesarias
# script_dir = os.path.dirname(__file__)
# logic_dir = os.path.join(script_dir, ".", "logic")
# sys.path.append(logic_dir)

from logic.logic import RungeKutta  # Importa RungeKutta
from logic.error import (
    Parentesis_Error,
    RK_Error,
    Inf,
)  # Importa las excepciones personalizadas

st.subheader("Graficadora")

# División de columnas para la interfaz
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
                print("ast parseado:", rk_solver.ast)
                X, Y = rk_solver.solver()
                with plot_col:
                    # Graficar la solución con Seaborn
                    sns.set(style="whitegrid")
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(x=X, y=Y, label="Solución RK", marker="o")
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt.gcf())
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
