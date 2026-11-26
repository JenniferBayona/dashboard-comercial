"""Carga y segmentación descriptiva de pedidos, sin volver a ajustar modelos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def cargar_datos(raiz):
    raiz = Path(raiz)
    df = pd.read_csv(raiz / "data/dataset_procesado.csv", parse_dates=["order_date"])
    segmentos = pd.read_csv(raiz / "data/segmentos_kmeans.csv")
    if df["indice_origen"].duplicated().any() or segmentos["indice_origen"].duplicated().any():
        raise ValueError("indice_origen debe ser único para unir los pedidos con sus clústeres.")
    if df["order_id"].duplicated().any() or df["order_date"].isna().any():
        raise ValueError("Se encontraron pedidos duplicados o fechas inválidas.")
    usd = df.loc[df["currency"].eq("USD")].set_index("indice_origen")
    seg = segmentos.set_index("indice_origen")
    if set(usd.index) != set(seg.index):
        raise ValueError("Los segmentos deben corresponder exactamente a los pedidos en USD.")
    for col in ["customer_age", "quantity", "net_sales"]:
        if not np.allclose(usd.loc[seg.index, col], seg[col], rtol=1e-9, atol=1e-9):
            raise ValueError(f"La unión de segmentos no coincide para {col}.")
    columnas = ["indice_origen", "cluster_kmeans", "perfil_comercial"]
    df = df.merge(segmentos[columnas], on="indice_origen", how="left", validate="one_to_one")
    if df.loc[df.currency.ne("USD"), "cluster_kmeans"].notna().any():
        raise ValueError("Se asignaron clústeres a una moneda diferente de USD.")
    df["cluster_kmeans"] = df["cluster_kmeans"].astype("Int64")
    return df


def cargar_metricas(raiz):
    ruta = Path(raiz) / "reports"
    return {
        "regresion": pd.read_csv(ruta / "metricas_regresion.csv", index_col=0),
        "clasificacion": pd.read_csv(ruta / "metricas_clasificacion.csv", index_col=0),
        "clustering": pd.read_csv(ruta / "metricas_clustering.csv"),
        "informacion": json.loads((ruta / "informacion_modelos.json").read_text(encoding="utf-8")),
    }


def filtrar_pedidos(df, moneda, fechas, categorias=None, edad=None, cantidad=None, ventas=None):
    """Categorías vacías significan cero resultados; se utiliza una sola moneda."""
    mask = df["currency"].eq(moneda)
    if fechas is not None:
        inicio, fin = pd.Timestamp(fechas[0]), pd.Timestamp(fechas[1])
        mask &= df["order_date"].between(inicio, fin)
    for columna, valores in (categorias or {}).items():
        mask &= df[columna].isin(valores)
    for columna, rango in [("customer_age", edad), ("quantity", cantidad), ("net_sales", ventas)]:
        if rango is not None:
            mask &= df[columna].between(rango[0], rango[1])
    return df.loc[mask].copy()


def calcular_kpis(df):
    monedas = df["currency"].dropna().unique()
    if len(monedas) > 1:
        raise ValueError("Los importes deben calcularse para una sola moneda.")
    finalizados = df["order_status"].isin(["Completed", "Returned"])
    completados = df.loc[df["order_status"].eq("Completed")]
    tasa = float(df.loc[finalizados, "order_status"].eq("Returned").mean()) if finalizados.any() else None
    return {
        "pedidos": int(len(df)),
        "clientes": int(df["customer_id"].nunique()),
        "ventas_completadas": float(completados["net_sales"].sum()),
        "ticket_completado": float(completados["net_sales"].mean()) if len(completados) else None,
        "tasa_devolucion": tasa,
        "pedidos_finalizados": int(finalizados.sum()),
    }


def perfiles_clusters(df):
    datos = df.loc[df["currency"].eq("USD") & df["cluster_kmeans"].notna()]
    return (datos.groupby(["cluster_kmeans", "perfil_comercial"], observed=True)
            .agg(pedidos=("order_id", "size"), edad_promedio=("customer_age", "mean"),
                 unidades_promedio=("quantity", "mean"), venta_promedio_usd=("net_sales", "mean"))
            .reset_index())
