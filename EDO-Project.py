import streamlit as st
import os


st.set_page_config(
    layout="wide"
)  # Asegúrate de que esto sea el primer comando de Streamlit

st.title("Proyecto de Ecuaciones Diferenciales Ordinarias")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Menu")
    option = st.radio("Selecciona una opción:", ["Graficadora", "Sistemas de Ecuaciones", "Informe", "Acerca de"])

if option == "Graficadora":
    # Leer el contenido de graficadora.py y ejecutarlo
    with open(os.path.join("UI", "graficadora.py"), "r", encoding="utf-8") as file:
        exec(file.read())
elif option == "Sistemas de Ecuaciones":
    # Leer el contenido de sequations.py y ejecutarlo
    with open(os.path.join("UI", "sequations.py"), "r", encoding="utf-8") as file:
        exec(file.read())
elif option == "Informe":
    st.subheader("Informe")
elif option == "Acerca de":
    st.subheader("Acerca de")
