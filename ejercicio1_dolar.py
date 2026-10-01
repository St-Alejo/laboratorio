"""Ejercicio 1: Predicción del precio del dólar con regresión lineal múltiple."""
import comun as c

NOMBRE = "dolar"
ARCHIVO = "dolar_data.csv"
FEATURES = ["Dia", "Inflacion", "Tasa_interes"]
OBJETIVO = "Precio_Dolar"
UNIDAD = "pesos"


def main():
    # Fase 1: Comprensión del negocio
    c.titulo("EJERCICIO 1 - PRECIO DEL DÓLAR | FASE 1 - COMPRENSIÓN DEL NEGOCIO")
    print("Objetivo: estimar el precio del dólar a partir del día, la inflación y la tasa de interés.")

    # Fases 2 y 3: datos
    df = c.cargar_datos(ARCHIVO)
    c.explorar(df, OBJETIVO)
    df, X_train, X_test, y_train, y_test = c.preparar(df, FEATURES, OBJETIVO)

    # Fase 4: modelado e interpretación de coeficientes
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
