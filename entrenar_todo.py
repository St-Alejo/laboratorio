"""Ejecuta los tres ejercicios y muestra un resumen comparativo de métricas."""
import ejercicio1_dolar
import ejercicio2_glucosa
import ejercicio3_energia

if __name__ == "__main__":
    resumen = {
        "Dólar": ejercicio1_dolar.main(),
        "Glucosa": ejercicio2_glucosa.main(),
        "Energía": ejercicio3_energia.main(),
    }
    print("\n" + "=" * 70 + "\nRESUMEN DE LOS TRES MODELOS\n" + "=" * 70)
    for nombre, m in resumen.items():
        print(f"{nombre:<8} MSE={m['MSE']:>10.4f}  RMSE={m['RMSE']:>8.4f}  R2={m['R2']:.4f}")
