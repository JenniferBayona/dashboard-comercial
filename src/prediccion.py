"""Inferencia con los pipelines y transformadores exportados en la Fase 3."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn


def cargar_modelos(raiz):
    if sklearn.__version__ != "1.6.1":
        raise RuntimeError("Instala scikit-learn==1.6.1 antes de cargar los modelos de Fase 3.")
    ruta = Path(raiz) / "models"
    # Estos archivos proceden del ZIP del proyecto y son de confianza.
    return {
        "regresion": joblib.load(ruta / "modelo_regresion.joblib"),
        "clasificacion": joblib.load(ruta / "modelo_clasificacion.joblib"),
        "clustering": joblib.load(ruta / "modelo_clustering.joblib"),
    }


def construir_entrada(valores, fecha):
    """Deriva mes y fin de semana con la misma definición de la Fase 2."""
    fila = dict(valores)
    fecha = pd.Timestamp(fecha)
    fila["order_month"] = int(fecha.month)
    fila["is_weekend"] = int(fecha.dayofweek >= 5)
    if not 18 <= int(fila["customer_age"]) <= 74:
        raise ValueError("La edad debe estar entre 18 y 74 años, el rango observado.")
    if not 1 <= int(fila["quantity"]) <= 24:
        raise ValueError("La cantidad debe estar entre 1 y 24, el rango observado.")
    return pd.DataFrame([fila])


def predecir_pedido(modelos, entrada):
    clf = modelos["clasificacion"]
    pipeline = clf["pipeline"]
    indice_positivo = int(np.flatnonzero(np.asarray(pipeline.classes_) == 1)[0])
    score = float(pipeline.predict_proba(entrada[clf["variables"]])[0, indice_positivo])
    umbral = float(clf["umbral"])
    resultado = {
        "puntuacion_devolucion": score,
        "umbral": umbral,
        "alerta_exploratoria": bool(score >= umbral),
        "venta_estimada_usd": None,
    }
    if entrada.iloc[0]["currency"] == "USD":
        reg = modelos["regresion"]
        resultado["venta_estimada_usd"] = float(reg["pipeline"].predict(entrada[reg["variables"]])[0])
    return resultado


def proyectar_clusters(modelos, df):
    """Transforma y proyecta sin fit: conserva la geometría de la Fase 3."""
    cluster = modelos["clustering"]
    if "currency" in df and not df["currency"].eq("USD").all():
        raise ValueError("La segmentación está definida exclusivamente para USD.")
    if df["quantity"].le(0).any() or df["net_sales"].lt(0).any():
        raise ValueError("La segmentación necesita cantidades positivas y ventas no negativas.")
    x = pd.DataFrame({
        "customer_age": df["customer_age"],
        "log_quantity": np.log1p(df["quantity"]),
        "log_net_sales": np.log1p(df["net_sales"]),
    }, index=df.index)
    x_std = cluster["escalador"].transform(x[cluster["variables"]])
    etiquetas = cluster["modelo"].predict(x_std)
    pca = cluster["pca"].transform(x_std)
    salida = df.copy()
    salida["cluster_kmeans"] = etiquetas
    salida["perfil_comercial"] = pd.Series(etiquetas, index=df.index).map(cluster["nombres_clusters"])
    salida["PCA 1"] = pca[:, 0]
    salida["PCA 2"] = pca[:, 1]
    return salida
