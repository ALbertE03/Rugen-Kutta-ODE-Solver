import streamlit as st
import os


st.set_page_config(layout="wide")

st.title("Proyecto de Ecuaciones Diferenciales Ordinarias")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Menu")
    option = st.radio(
        "Selecciona una opción:",
        ["Graficadora", "Sistemas de Ecuaciones", "Orden Superior"],
    )

if option == "Graficadora":
    with open(os.path.join("UI", "graficadora.py"), "r", encoding="utf-8") as file:
        exec(file.read())
elif option == "Sistemas de Ecuaciones":
    with open(os.path.join("UI", "sequations.py"), "r", encoding="utf-8") as file:
        exec(file.read())
elif option == "Orden Superior":
    with open(os.path.join("UI", "orden_sup.py"), "r", encoding="utf-8") as file:
        exec(file.read())
