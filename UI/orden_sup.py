import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from logic_sup import Sup

sup = Sup()

st.title("Solver de Ecuaciones Diferenciales Ordinarias de Orden n (1D)")

st.markdown("""
Este solver permite ingresar una EDO lineal de la forma:

\\[
   a_n \\frac{d^n y}{dx^n} \\,+\\, a_{n-1} \\frac{d^{n-1} y}{dx^{n-1}} 
   \\,+\\, \\cdots \\,+\\, a_1 \\frac{dy}{dx} \\,+\\, a_0 y 
   \\;=\\; f(x).
\\]

""")

order = st.number_input("Orden de la EDO (n)", min_value=1, max_value=10, value=2, step=1)

HOMOG = st.checkbox("¿Es homogenea?", value=True)

st.markdown("---")

coefs = []
for i in range(order, -1, -1):
    default_val = 1.0 if i == order else 0.0
    val = st.number_input(f"Coeficiente a_{i}", value=default_val, step=1.0, format="%.2f")
    coefs.append(val)

f_expr = "0"
if not HOMOG:
    f_expr = st.text_input("Término no-homogéneo f(x)", "sin(x)")

st.markdown("---")

cond_iniciales = []
st.markdown("### Condiciones iniciales")
for i in range(order):
    val_ci = st.number_input(f"y^{ '('+str(i)+' veces' if i>0 else '' }(0) : Orden {i}", value=1.0, step=1.0, format="%.2f")
    cond_iniciales.append(val_ci)

if st.button("Resolver EDO"):
    array = []
    # Build the list of (coef, derivative_order) for the homogeneous part
    # Example: if order=2 and coefs=[a2,a1,a0], we get
    # [(a2,2), (a1,1), (a0,0)] for the homogeneous part.
    for i in range(order+1):
        grado = order - i
        array.append((coefs[i], grado))

    # If it's non-homogeneous, the last element in `array` is the forcing term f(x)
    # instead of a coefficient. But the logic_sup approach requires us to place f(x)
    # at the *end* of the array. So we remove the final tuple and replace it with f(x).
    if not HOMOG:
        array_non_hom = array[:-1]  # keep all except the last
        # the last element is the symbolic expression for f(x)
        array_non_hom.append(sp.sympify(f_expr))
        array = array_non_hom

    sol_tex, sol_numeric = sup.get_solution(array, cond_iniciales, HOMOG)

    st.markdown("### Solución simbólica")
    st.latex(sol_tex)

    st.markdown("### Gráfica de la solución numérica (aprox)")
    xs = np.linspace(0, 10, 200)
    ys = [sol_numeric(x_val) for x_val in xs]
    fig, ax = plt.subplots()
    ax.plot(xs, ys, label="Solución")
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.legend()
    st.pyplot(fig)
