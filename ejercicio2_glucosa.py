"""Ejercicio 2: Predicción del nivel de glucosa en sangre con regresión lineal múltiple."""
import comun as c

NOMBRE = "glucosa"
ARCHIVO = "glucosa_data.csv"
FEATURES = ["Edad", "IMC", "Actividad_Fisica"]
OBJETIVO = "Nivel_Glucosa"
UNIDAD = "mg/dL"


def main():
    # Fase 1: Comprensión del negocio
    c.titulo("EJERCICIO 2 - GLUCOSA EN SANGRE | FASE 1 - COMPRENSIÓN DEL NEGOCIO")
    print("Objetivo: estimar la glucosa (mg/dL) según edad, IMC y horas semanales de ejercicio.")

    # Fases 2 y 3: datos
    df = c.cargar_datos(ARCHIVO)
    c.explorar(df, OBJETIVO)
    df, X_train, X_test, y_train, y_test = c.preparar(df, FEATURES, OBJETIVO)

    # Fase 4: modelado, coeficientes e importancia (variable de mayor impacto)
    modelo = c.entrenar(X_train, y_train)
    c.interpretar(modelo, FEATURES, OBJETIVO, UNIDAD)
    tabla = c.importancia(modelo, X_train, y_train, FEATURES)

    # Fase 5: evaluación (MSE y R²) y gráficas
    metricas, y_pred = c.evaluar(modelo, X_test, y_test)
    c.graficar_relaciones(df, FEATURES, OBJETIVO, NOMBRE)
    c.graficar_correlacion(df, NOMBRE)
    c.graficar_real_vs_pred(y_test, y_pred, OBJETIVO, NOMBRE)

    # Fase 6: exportación
    c.exportar(modelo, FEATURES, OBJETIVO, metricas, NOMBRE, X_test)
    c.guardar_resultados(NOMBRE, metricas, tabla, modelo)
    return metricas


if __name__ == "__main__":
    main()
