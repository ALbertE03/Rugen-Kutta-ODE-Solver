import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import altair as alt

from logic.logic_sup import Sup

sup = Sup()

st.title("EDO de Orden Superior")

left_col, right_col = st.columns([1, 2])

with left_col:
    tipo_edo = st.radio("Tipo de Ecuación:", ("Ecuación Homogénea", "Ecuación No Homogénea"))
    order = st.number_input("Orden (n)", min_value=1, max_value=10, value=2, step=1)

    coefs = []
    for i in range(order, -1, -1):
        val = st.number_input(f"Coeficiente a_{i}", value=(1.0 if i == order else 0.0), format="%.2f")
        coefs.append(val)

    f_expr = "0"
    if tipo_edo == "Ecuación No Homogénea":
        f_expr = st.text_input("f(x)", "sin(x)")

    cond_iniciales = []
    for i in range(order):
        ci_val = st.number_input(
            f"Condición inicial (derivada de orden {i} en x=0)",
            value=1.0, format="%.2f"
        )
        cond_iniciales.append(ci_val)

    boton_resolver = st.button("Resolver")

with right_col:
    if boton_resolver:
        array = []
        for i in range(order + 1):
            grado = order - i
            array.append((coefs[i], grado))

        HOMOG = (tipo_edo == "Ecuación Homogénea")

        if not HOMOG:
            array_non_hom = array[:-1]
            array_non_hom.append(sp.sympify(f_expr))
            array = array_non_hom

        sol_tex, sol_func = sup.get_solution(array, cond_iniciales, HOMOG)

        st.markdown("### Gráfica de la solución")
        xs = np.linspace(0, 10, 200)
        ys = [sol_func(xv) for xv in xs]

        df = pd.DataFrame({"x": xs, "y": ys})

        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("x", title="x"),
                y=alt.Y("y", title="y(x)"),
                tooltip=["x", "y"]
            )
            .interactive()
            .properties(width="container", height=400)
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("### Solución simbólica")
        st.latex(sol_tex)
