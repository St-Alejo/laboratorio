"""Funciones compartidas por los tres ejercicios, organizadas según las fases de CRISP-DM."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Rutas del proyecto (relativas a este archivo) ---
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELOS = BASE / "modelos"
GRAFICAS = BASE / "graficas"
RESULTADOS = BASE / "resultados"
for carpeta in (MODELOS, GRAFICAS, RESULTADOS):
    carpeta.mkdir(exist_ok=True)


def titulo(texto):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


# --- Fase 2: Comprensión de los datos ---
def cargar_datos(nombre_csv):
    return pd.read_csv(DATA / nombre_csv)


def explorar(df, objetivo):
    titulo("FASE 2 - COMPRENSIÓN DE LOS DATOS")
    print(f"Filas y columnas: {df.shape}")
    print(f"Valores nulos: {df.isna().sum().sum()} | Filas duplicadas: {df.duplicated().sum()}")
    print("\nEstadística descriptiva:")
    print(df.describe().T.round(3).to_string())
    print(f"\nCorrelación de Pearson con {objetivo}:")
    print(df.corr()[objetivo].drop(objetivo).round(4).to_string())


# --- Fase 3: Preparación de los datos ---
def preparar(df, features, objetivo):
    titulo("FASE 3 - PREPARACIÓN DE LOS DATOS")
    df = df.dropna().drop_duplicates()
    X, y = df[features], df[objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas (80/20)")
    return df, X_train, X_test, y_train, y_test


# --- Fase 4: Modelado ---
def entrenar(X_train, y_train):
    titulo("FASE 4 - MODELADO (Regresión lineal múltiple)")
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo


def interpretar(modelo, features, objetivo, unidad):
    print(f"Intercepto (b0): {modelo.intercept_:.4f}")
    for var, coef in zip(features, modelo.coef_):
        efecto = "aumenta" if coef > 0 else "disminuye"
        print(f"  b({var}) = {coef:+.4f} -> por cada unidad que sube {var}, "
              f"{objetivo} {efecto} {abs(coef):.4f} {unidad} (lo demás constante)")
    terminos = " ".join(f"{c:+.4f}*{v}" for v, c in zip(features, modelo.coef_))
    print(f"\nEcuación: {objetivo} = {modelo.intercept_:.4f} {terminos}")


def importancia(modelo, X, y, features):
    tabla = pd.DataFrame({
        "Variable": features,
        "Coeficiente": modelo.coef_,
        "Coef_estandarizado": modelo.coef_ * X[features].std().values / y.std(),
    })
    tabla["Impacto_abs"] = tabla["Coef_estandarizado"].abs()
    tabla = tabla.sort_values("Impacto_abs", ascending=False).reset_index(drop=True)
    print("\nImportancia de variables (coeficientes estandarizados, mayor a menor):")
    print(tabla.round(4).to_string(index=False))
    print(f"\n>> Variable con MAYOR impacto: {tabla.loc[0, 'Variable']}")
    return tabla


# --- Fase 5: Evaluación ---
def evaluar(modelo, X_test, y_test):
    titulo("FASE 5 - EVALUACIÓN")
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metricas = {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    for nombre, valor in metricas.items():
        print(f"  {nombre:<5}: {valor:.4f}")
    return metricas, y_pred


# --- Visualizaciones ---
def graficar_relaciones(df, features, objetivo, nombre):
    fig, ejes = plt.subplots(1, len(features), figsize=(6 * len(features), 5))
    for eje, var in zip(ejes, features):
        sns.regplot(data=df, x=var, y=objetivo, ax=eje,
                    scatter_kws={"alpha": 0.35, "s": 12}, line_kws={"color": "red"})
        eje.set_title(f"{var} vs {objetivo} (r = {df[var].corr(df[objetivo]):.3f})")
    fig.tight_layout()
    fig.savefig(GRAFICAS / f"{nombre}_relaciones.png", dpi=120)
    plt.close(fig)


def graficar_correlacion(df, nombre):
    fig, eje = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=eje)
    eje.set_title(f"Matriz de correlación - {nombre}")
    fig.tight_layout()
    fig.savefig(GRAFICAS / f"{nombre}_correlacion.png", dpi=120)
    plt.close(fig)


def graficar_real_vs_pred(y_test, y_pred, objetivo, nombre):
    fig, eje = plt.subplots(figsize=(6, 6))
    eje.scatter(y_test, y_pred, alpha=0.4, s=12)
    limites = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    eje.plot(limites, limites, "r--", label="Predicción perfecta")
    eje.set_xlabel(f"{objetivo} real")
    eje.set_ylabel(f"{objetivo} predicho")
    eje.set_title(f"Real vs Predicho - {nombre}")
    eje.legend()
    fig.tight_layout()
    fig.savefig(GRAFICAS / f"{nombre}_real_vs_pred.png", dpi=120)
    plt.close(fig)


# --- Fase 6: Despliegue (exportar el modelo) ---
def exportar(modelo, features, objetivo, metricas, nombre, X_test):
    titulo("FASE 6 - DESPLIEGUE (exportar modelo)")
    ruta = MODELOS / f"{nombre}.joblib"
    paquete = {"modelo": modelo, "features": features, "objetivo": objetivo,
               "metricas": {k: float(v) for k, v in metricas.items()}}
    joblib.dump(paquete, ruta)
    cargado = joblib.load(ruta)
    iguales = np.allclose(cargado["modelo"].predict(X_test), modelo.predict(X_test))
    print(f"Modelo guardado en: {ruta.relative_to(BASE)}")
    print(f"Verificación al recargar (predicciones idénticas): {iguales}")


def guardar_resultados(nombre, metricas, tabla_importancia, modelo):
    tabla_importancia.to_csv(RESULTADOS / f"coeficientes_{nombre}.csv", index=False)
    with open(RESULTADOS / f"metricas_{nombre}.txt", "w", encoding="utf-8") as f:
        f.write(f"Intercepto: {modelo.intercept_:.6f}\n")
        for k, v in metricas.items():
            f.write(f"{k}: {v:.6f}\n")
