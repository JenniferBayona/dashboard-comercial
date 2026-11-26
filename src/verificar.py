"""Comprueba integridad, inferencia y reproducción de métricas de prueba."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.datos import cargar_datos, cargar_metricas, calcular_kpis, filtrar_pedidos
from src.prediccion import cargar_modelos, proyectar_clusters, construir_entrada, predecir_pedido


def verificar(raiz=RAIZ):
    raiz = Path(raiz)
    df = cargar_datos(raiz)
    modelos = cargar_modelos(raiz)
    metricas = cargar_metricas(raiz)
    usd = df.loc[df.currency.eq("USD")].copy()
    supervisados = df.loc[df.order_status.isin(["Completed", "Returned"])].copy()
    assert len(df) == 138116 and len(usd) == 82600
    assert len(supervisados) == 123021
    proyeccion = proyectar_clusters(modelos, usd)
    assert np.array_equal(proyeccion.cluster_kmeans, usd.cluster_kmeans.astype(int))
    assert (proyeccion.perfil_comercial == usd.perfil_comercial).all()
    assert np.isfinite(proyeccion[["PCA 1", "PCA 2"]].to_numpy()).all()
    assert filtrar_pedidos(df, "USD", None, {"sales_channel": []}).empty
    assert calcular_kpis(df.loc[df.order_status.eq("Pending") & df.currency.eq("USD")])["tasa_devolucion"] is None
    try:
        calcular_kpis(df)
    except ValueError:
        pass
    else:
        raise AssertionError("No se deben sumar monedas distintas.")

    # Mismos splits de Fase 2. Solo se predice: no se ajustan ni seleccionan modelos.
    _, temp_reg = train_test_split(usd, test_size=0.30, random_state=42)
    _, test_reg = train_test_split(temp_reg, test_size=0.50, random_state=42)
    reg = modelos["regresion"]
    pred_reg = reg["pipeline"].predict(test_reg[reg["variables"]])
    calculadas_reg = {
        "MAE": mean_absolute_error(test_reg.net_sales, pred_reg),
        "RMSE": float(np.sqrt(mean_squared_error(test_reg.net_sales, pred_reg))),
        "R2": r2_score(test_reg.net_sales, pred_reg),
    }
    for nombre, valor in calculadas_reg.items():
        assert np.isclose(valor, metricas["regresion"].loc["Prueba", nombre], rtol=1e-8, atol=1e-8), (nombre, valor)

    y = supervisados.order_status.eq("Returned").astype(int)
    _, temp_clf = train_test_split(supervisados, test_size=0.30, stratify=y, random_state=42)
    _, test_clf = train_test_split(temp_clf, test_size=0.50,
                                 stratify=temp_clf.order_status.eq("Returned").astype(int), random_state=42)
    clf = modelos["clasificacion"]
    clases = np.asarray(clf["pipeline"].classes_)
    indice_positivo = int(np.flatnonzero(clases == 1)[0])
    scores = clf["pipeline"].predict_proba(test_clf[clf["variables"]])[:, indice_positivo]
    pred_clf = scores >= clf["umbral"]
    real_clf = test_clf.order_status.eq("Returned").astype(int)
    calculadas_clf = {
        "Accuracy": accuracy_score(real_clf, pred_clf),
        "Precision": precision_score(real_clf, pred_clf, zero_division=0),
        "Recall": recall_score(real_clf, pred_clf, zero_division=0),
        "F1-Score": f1_score(real_clf, pred_clf, zero_division=0),
        "AUC-ROC": roc_auc_score(real_clf, scores),
        "Average Precision": average_precision_score(real_clf, scores),
    }
    for nombre, valor in calculadas_clf.items():
        assert np.isclose(valor, metricas["clasificacion"].loc["Prueba", nombre], rtol=1e-8, atol=1e-8), (nombre, valor)

    entrada = construir_entrada(usd.iloc[0].to_dict(), usd.iloc[0].order_date)
    demo = predecir_pedido(modelos, entrada)
    assert demo["venta_estimada_usd"] is not None
    assert 0 <= demo["puntuacion_devolucion"] <= 1
    entrada["currency"] = "EUR"
    assert predecir_pedido(modelos, entrada)["venta_estimada_usd"] is None
    assert np.isclose(modelos["clasificacion"]["umbral"], metricas["informacion"]["umbral_clasificacion"])

    resultado = {
        "integridad_datos": "correcta", "registros": int(len(df)), "pedidos_usd": int(len(usd)),
        "clusters_coincidentes": int(len(proyeccion)), "metricas_prueba_reproducidas": True,
        "regresion_prueba": calculadas_reg, "clasificacion_prueba": calculadas_clf,
        "filas_prueba_regresion": int(len(test_reg)), "filas_prueba_clasificacion": int(len(test_clf)),
        "umbral_clasificacion": float(clf["umbral"]),
        "publicacion_dashboard": "pendiente de publicar y verificar URL",
    }
    (raiz / "reports/verificacion.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


if __name__ == "__main__":
    print(json.dumps(verificar(), ensure_ascii=False, indent=2))
