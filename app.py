"""Interfaz web (Streamlit) para predecir con los tres modelos exportados."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELOS = Path(__file__).resolve().parent / "modelos"

# --- Configuración de cada ejercicio: archivo, unidad y rango de cada entrada ---
EJERCICIOS = {
    "Dólar": {
        "archivo": "dolar.joblib",
        "unidad": "pesos",
        "descripcion": "Predice el precio del dólar según el día, la inflación y la tasa de interés.",
        "entradas": {
            "Dia": dict(label="Día (número)", min_value=1, max_value=1000, value=250, step=1),
            "Inflacion": dict(label="Inflación diaria (ej: 0.02)", min_value=0.0, max_value=0.1,
                              value=0.02, step=0.001, format="%.4f"),
            "Tasa_interes": dict(label="Tasa de interés diaria (%)", min_value=0.0, max_value=20.0,
                                 value=5.0, step=0.1),
        },
    },
    "Glucosa": {
        "archivo": "glucosa.joblib",
        "unidad": "mg/dL",
        "descripcion": "Predice el nivel de glucosa en sangre según edad, IMC y actividad física.",
        "entradas": {
            "Edad": dict(label="Edad (años)", min_value=1, max_value=120, value=45, step=1),
            "IMC": dict(label="Índice de masa corporal", min_value=10.0, max_value=60.0,
                        value=25.0, step=0.1),
            "Actividad_Fisica": dict(label="Actividad física (horas/semana)", min_value=0,
                                     max_value=40, value=4, step=1),
        },
    },
    "Energía": {
        "archivo": "energia.joblib",
        "unidad": "kWh",
        "descripcion": "Predice el consumo eléctrico según temperatura, hora y día de la semana.",
        "entradas": {
            "Temperatura": dict(label="Temperatura (°C)", min_value=-10.0, max_value=50.0,
                                value=25.0, step=0.5),
            "Hora": dict(label="Hora del día (1 a 24)", min_value=1, max_value=24, value=12, step=1),
            "Dia_Semana": dict(label="Día de la semana (1=Lunes ... 7=Domingo)", min_value=1,
                               max_value=7, value=1, step=1),
        },
    },
}


# --- Carga del modelo (se guarda en caché para no leer el archivo en cada interacción) ---
@st.cache_resource
def cargar_modelo(archivo):
    return joblib.load(MODELOS / archivo)


# --- Encabezado y selector de ejercicio ---
st.set_page_config(page_title="Predicciones - Regresión Lineal", page_icon="📈", layout="centered")
st.title(" Predicciones con Regresión Lineal Múltiple")
st.caption("Laboratorio 1 - Minería de Datos (CRISP-DM)")

opcion = st.sidebar.selectbox("Selecciona el ejercicio", list(EJERCICIOS))
config = EJERCICIOS[opcion]
st.subheader(f"Ejercicio: {opcion}")
st.write(config["descripcion"])

ruta = MODELOS / config["archivo"]
if not ruta.exists():
    st.error(f"No se encontró {ruta.name}. Ejecuta primero: python entrenar_todo.py")
    st.stop()

paquete = cargar_modelo(config["archivo"])
modelo, features = paquete["modelo"], paquete["features"]

# --- Formulario: el usuario ingresa los valores por teclado ---
with st.form("formulario"):
    valores = {var: st.number_input(**config["entradas"][var]) for var in features}
    enviar = st.form_submit_button("Predecir")

# --- Predicción y resultado ---
if enviar:
    entrada = pd.DataFrame([valores], columns=features)
    prediccion = float(modelo.predict(entrada)[0])
    st.metric(f"{paquete['objetivo']} estimado", f"{prediccion:,.2f} {config['unidad']}")

    if opcion == "Glucosa":
        if prediccion < 100:
            st.success("Rango de referencia: normal (< 100 mg/dL en ayunas).")
        elif prediccion < 126:
            st.warning("Rango de referencia: prediabetes (100 - 125 mg/dL en ayunas).")
        else:
            st.error("Rango de referencia: diabetes (≥ 126 mg/dL en ayunas).")
        st.caption("Resultado académico, no es un diagnóstico médico.")

# --- Información del modelo: métricas y ecuación ---
with st.expander("Ver detalles del modelo"):
    m = paquete["metricas"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{m['R2']:.4f}")
    c2.metric("MSE", f"{m['MSE']:,.2f}")
    c3.metric("RMSE", f"{m['RMSE']:,.2f}")
    terminos = " ".join(f"{c:+.4f}·{v}" for v, c in zip(features, modelo.coef_))
    st.code(f"{paquete['objetivo']} = {modelo.intercept_:.4f} {terminos}")
