"""Ejecutar desde la raíz: python -m streamlit run dashboard/app.py"""
from pathlib import Path
from datetime import date
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.datos import cargar_datos, cargar_metricas, filtrar_pedidos, calcular_kpis, perfiles_clusters
from src.prediccion import cargar_modelos, construir_entrada, predecir_pedido, proyectar_clusters

st.set_page_config(page_title="Pedidos | Ciencia de Datos", page_icon="📊", layout="wide")
st.markdown("""<style>
.stApp {background: #f8fafc;}
h1, h2, h3 {color: #13243a;}
[data-testid="stMetric"] {background: white; padding: 16px; border-radius: 12px;
 border: 1px solid #e2e8f0;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def datos_dashboard():
    return cargar_datos(RAIZ)


@st.cache_data(show_spinner=False)
def metricas_dashboard():
    return cargar_metricas(RAIZ)


@st.cache_resource(show_spinner=False)
def modelos_dashboard():
    return cargar_modelos(RAIZ)


df = datos_dashboard()
metricas = metricas_dashboard()
st.title("Pedidos, ventas y perfiles de compra")
st.caption("Proyecto integrador · Jennifer Juliana Bayona Rojas · Universidad de La Salle")
st.info("Datos simulados de comercio electrónico. Los resultados describen este conjunto de datos.")

with st.sidebar:
    st.header("Explorar pedidos")
    st.caption("Estos filtros se aplican al resumen y los segmentos. El simulador tiene sus propias entradas.")
    monedas = ["USD"] + sorted(set(df.currency) - {"USD"})
    moneda = st.selectbox("Moneda de los importes", monedas, key="f_moneda")
    base_moneda = df.loc[df.currency.eq(moneda)]
    fecha_min, fecha_max = df.order_date.min().date(), df.order_date.max().date()
    fechas = st.date_input("Periodo de los pedidos", (fecha_min, fecha_max),
                           min_value=fecha_min, max_value=fecha_max, key="f_fechas")
    categorias = {}
    for columna, etiqueta in [("sales_channel", "Canal de venta"), ("customer_country", "País"),
                              ("customer_segment", "Segmento del cliente"), ("order_status", "Estado del pedido")]:
        opciones = sorted(df[columna].dropna().unique().tolist())
        categorias[columna] = st.multiselect(etiqueta, opciones, default=opciones, key="f_" + columna)
    edad = st.slider("Edad del cliente", 18, 74, (18, 74), key="f_edad")
    cantidad = st.slider("Unidades por pedido", 1, 24, (1, 24), key="f_cantidad")
    limite_venta = float(base_moneda.net_sales.max())
    ventas = st.slider(f"Importe registrado ({moneda})", 0.0, limite_venta, (0.0, limite_venta), key="f_ventas_" + moneda)

if len(fechas) != 2:
    st.warning("Completa las dos fechas del periodo para actualizar el resumen.")
    filtrado = df.iloc[:0].copy()
else:
    filtrado = filtrar_pedidos(df, moneda, fechas, categorias, edad, cantidad, ventas)

resumen_tab, pred_tab, cluster_tab, metodo_tab = st.tabs([
    "Resumen comercial", "Simular un pedido", "Segmentos de pedidos", "Resultados y alcance"
])

with resumen_tab:
    st.subheader(f"Resumen de pedidos en {moneda}")
    if filtrado.empty:
        st.info("No hay pedidos para esta combinación de filtros. Amplía el periodo o las categorías.")
    else:
        kpis = calcular_kpis(filtrado)
        cols = st.columns(5)
        cols[0].metric("Pedidos", f"{kpis['pedidos']:,}")
        cols[1].metric("Clientes distintos", f"{kpis['clientes']:,}")
        cols[2].metric(f"Venta de completados ({moneda})", f"{kpis['ventas_completadas']:,.2f}")
        ticket = kpis["ticket_completado"]
        cols[3].metric(f"Ticket de completados ({moneda})", "Sin datos" if ticket is None else f"{ticket:,.2f}")
        tasa = kpis["tasa_devolucion"]
        cols[4].metric("Devoluciones / finalizados", "Sin datos" if tasa is None else f"{tasa:.2%}")
        st.caption("Venta y ticket: solo pedidos Completed. Tasa: Returned / (Completed + Returned) dentro de los filtros. "
                   "Si filtras por estado, cambia también ese denominador. Los importes no representan beneficio ni flujo de caja.")
        completados = filtrado.loc[filtrado.order_status.eq("Completed")].copy()
        izq, der = st.columns([2, 1])
        if not completados.empty:
            completados["Mes"] = completados.order_date.dt.to_period("M").dt.to_timestamp()
            mensual = completados.groupby("Mes", as_index=False).net_sales.sum()
            izq.plotly_chart(px.line(mensual, x="Mes", y="net_sales", markers=True,
                                    title=f"Venta mensual de pedidos completados ({moneda})",
                                    labels={"net_sales": f"Venta ({moneda})"}), width="stretch")
        estados = filtrado.order_status.value_counts().rename_axis("Estado").reset_index(name="Pedidos")
        der.plotly_chart(px.bar(estados, x="Estado", y="Pedidos", title="Estados de los pedidos",
                               color="Estado"), width="stretch")
        st.markdown("**Preguntas que orientan la lectura**")
        st.write("¿Cómo varían los importes y el ticket por canal y periodo? ¿Qué cambia en la proporción "
                 "de devoluciones? Estas diferencias describen asociaciones; no prueban efectos causales.")
        vista = filtrado[["order_id", "order_date", "currency", "sales_channel", "customer_country",
                          "quantity", "net_sales", "order_status", "perfil_comercial"]]
        st.dataframe(vista.head(500), hide_index=True, width="stretch")
        st.caption("La tabla muestra como máximo 500 pedidos; la descarga contiene todos los pedidos filtrados.")
        st.download_button("Descargar pedidos filtrados", vista.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"pedidos_{moneda}.csv", mime="text/csv")

with pred_tab:
    st.subheader("Simular las características de un pedido")
    st.write("Completa los datos disponibles al registrar la compra. El mes y el indicador de fin de semana "
             "se calculan a partir de la fecha.")
    with st.form("form_prediccion"):
        a, b, c = st.columns(3)
        fecha_pred = a.date_input("Fecha del pedido", date(2025, 6, 15), key="p_fecha")
        edad_pred = b.number_input("Edad", 18, 74, 35, step=1, key="p_edad")
        cantidad_pred = c.number_input("Cantidad de unidades", 1, 24, 5, step=1, key="p_cantidad")
        valores = {"customer_age": int(edad_pred), "quantity": int(cantidad_pred)}
        campos = [("currency", "Moneda"), ("customer_segment", "Segmento del cliente"),
                  ("sales_channel", "Canal de venta"), ("customer_country", "País del cliente"),
                  ("payment_method", "Método de pago"), ("shipping_method", "Tipo de envío"),
                  ("marketing_channel", "Canal de marketing")]
        for i, (campo, etiqueta) in enumerate(campos):
            opciones = sorted(df[campo].dropna().unique().tolist())
            indice = opciones.index("USD") if campo == "currency" else 0
            valores[campo] = [a, b, c][i % 3].selectbox(etiqueta, opciones, index=indice, key="p_" + campo)
        enviado = st.form_submit_button("Obtener predicción", type="primary")
    if enviado:
        entrada = construir_entrada(valores, fecha_pred)
        resultado = predecir_pedido(modelos_dashboard(), entrada)
        if resultado["venta_estimada_usd"] is not None:
            st.metric("Venta neta estimada (USD)", f"{resultado['venta_estimada_usd']:,.2f}")
            mae = metricas["regresion"].loc["Prueba", "MAE"]
            st.caption(f"Error absoluto medio en prueba: {mae:,.2f} USD. Es un promedio histórico del error; "
                       "no es un intervalo de confianza para esta predicción.")
        else:
            st.info("El modelo de ventas se entrenó únicamente en USD. Para esta moneda se muestra la clasificación exploratoria.")
        st.metric("Puntuación del clasificador de devolución", f"{resultado['puntuacion_devolucion']:.4f}")
        etiqueta = "Alerta exploratoria" if resultado["alerta_exploratoria"] else "Sin alerta exploratoria"
        st.write(f"**{etiqueta}** · Umbral de Fase 3: {resultado['umbral']:.4f}")
        st.warning("La puntuación no es una probabilidad calibrada. En prueba, la precisión fue 8,4 % "
                   "y el AUC-ROC fue 0,561. Este clasificador tiene capacidad limitada y genera muchas falsas alertas.")
        if not fecha_min <= fecha_pred <= fecha_max:
            st.info("La fecha está fuera del periodo observado (2021–2025). La predicción temporal requiere validación adicional.")
        with st.expander("Ver entradas utilizadas"):
            st.dataframe(entrada, hide_index=True)

with cluster_tab:
    st.subheader("Cuatro perfiles de pedidos en USD")
    st.write("Los grupos describen pedidos, no clientes permanentes. Se asignan después de conocer la venta neta.")
    st.caption("Se usan edad, log(1 + unidades) y log(1 + venta neta), junto con el escalador, K-Means y PCA de Fase 3.")
    if moneda != "USD":
        st.info("Selecciona USD en los filtros para explorar los segmentos definidos en el proyecto.")
    elif filtrado.empty:
        st.info("No hay pedidos en USD para esta combinación de filtros.")
    else:
        nombres = sorted(filtrado.perfil_comercial.dropna().unique().tolist())
        seleccion = st.multiselect("Perfiles que quieres visualizar", nombres, default=nombres, key="c_perfiles")
        grupos = filtrado.loc[filtrado.perfil_comercial.isin(seleccion)].copy()
        if grupos.empty:
            st.info("Selecciona al menos un perfil para mostrar el mapa.")
        else:
            perfiles = perfiles_clusters(grupos)
            st.dataframe(perfiles.round(2), hide_index=True, width="stretch")
            muestra = grupos.sample(n=min(5000, len(grupos)), random_state=42)
            proyectados = proyectar_clusters(modelos_dashboard(), muestra)
            fig = px.scatter(proyectados, x="PCA 1", y="PCA 2", color="perfil_comercial",
                             hover_data=["customer_age", "quantity", "net_sales", "order_date"],
                             title="Mapa interactivo de perfiles de pedidos", opacity=0.55,
                             labels={"perfil_comercial": "Perfil", "customer_age": "Edad",
                                     "quantity": "Unidades", "net_sales": "Venta (USD)"})
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Mapa: muestra reproducible de {len(muestra):,} de {len(grupos):,} pedidos seleccionados. "
                       "La tabla de perfiles usa todos los pedidos seleccionados. PCA conserva el 89,85 % de la variabilidad.")
            acciones = pd.DataFrame([
                ["Jóvenes - alto valor", "Evaluar paquetes de productos complementarios"],
                ["Mayores - menor valor", "Evaluar un producto complementario a precio regular"],
                ["Mayores - alto valor", "Evaluar un beneficio para la compra siguiente"],
                ["Jóvenes - menor valor", "Evaluar un combo de entrada de bajo precio"],
            ], columns=["Perfil del pedido", "Hipótesis para un piloto"])
            st.dataframe(acciones, hide_index=True, width="stretch")
            st.caption("Las acciones requieren un piloto con datos reales. No se ha medido un aumento de ventas.")

with metodo_tab:
    st.subheader("Rendimiento del modelo y alcance de la solución")
    st.caption("Métricas guardadas en Fase 3. Son resultados de las particiones originales y no cambian con los filtros.")
    r = metricas["regresion"].loc["Prueba"]
    c = metricas["clasificacion"].loc["Prueba"]
    col = st.columns(4)
    col[0].metric("RMSE de regresión (USD)", f"{r['RMSE']:,.2f}")
    col[1].metric("R² de regresión", f"{r['R2']:.3f}")
    col[2].metric("AUC-ROC de clasificación", f"{c['AUC-ROC']:.3f}")
    col[3].metric("F1 de clasificación", f"{c['F1-Score']:.3f}")
    st.markdown("**Regresión: Random Forest optimizado**")
    st.dataframe(metricas["regresion"].round(4), width="stretch")
    st.markdown("**Clasificación exploratoria: Random Forest y umbral fijado en validación**")
    st.dataframe(metricas["clasificacion"].round(4), width="stretch")
    st.markdown("**Comparación de segmentaciones**")
    st.dataframe(metricas["clustering"].round(4), hide_index=True, width="stretch")
    st.caption("K-Means se ajustó con 82.600 pedidos en USD y DBSCAN con una muestra de 8.000. "
               "Las siluetas exportadas se comparan sobre las mismas 7.932 filas sin ruido de DBSCAN.")
    st.write("La partición supervisada original es 70 % entrenamiento, 15 % validación y 15 % prueba; "
             "la clasificación es estratificada. La separación se hizo por pedido: hay clientes "
             "compartidos entre particiones y falta una evaluación temporal y por cliente.")
    st.write("La aplicación reutiliza los pipelines exportados. No reajusta modelos al cambiar filtros. "
             "Se excluyeron variables que revelaban el resultado, como entrega, reseña y puntos de fidelización.")
    st.link_button("Consultar el dataset de origen", "https://www.kaggle.com/datasets/datascikhan/e-commerce-sales-and-customer-analytics")
