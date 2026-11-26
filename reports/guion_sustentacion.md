# Guion de sustentación — 12 minutos

| Tiempo | Contenido | Qué mostrar |
|---|---|---|
| 0:00–1:00 | Problema, objetivo y unidad de análisis | Un pedido por fila; alcance exploratorio |
| 1:00–2:15 | Fuente y calidad de datos | Datos simulados, 138.116 pedidos y alcance USD |
| 2:15–3:30 | Preparación y prevención de fuga | Variables excluidas y división 70/15/15 |
| 3:30–5:00 | Modelos y resultados | Regresión R² 0,382; clasificación AUC 0,561 y precisión 8,4 % |
| 5:00–6:00 | Segmentación | K-Means K=4 frente a DBSCAN; perfiles de pedidos |
| 6:00–7:30 | Demostración de filtros y KPI | Abrir URL, cambiar periodo y canal; explicar denominadores |
| 7:30–9:00 | Demostración de predicción | Un pedido USD; interpretar error y puntuación exploratoria |
| 9:00–10:00 | Demostración de clústeres | Mapa PCA, filtro de perfil y promedios |
| 10:00–11:00 | Límites y trabajo futuro | Datos simulados, clientes compartidos y evaluación temporal |
| 11:00–12:00 | Conclusiones y reproducibilidad | Repo público, dependencias y propuestas de piloto |

## Preparar la demostración

Abre la URL pública antes de comenzar. Usa primero USD, un periodo amplio y un canal.
En el formulario prueba 35 años, 5 unidades y una fecha del periodo 2021–2025.
Explica que el clasificador usa el umbral ya seleccionado, sin reajustarlo en prueba.
Muestra el mapa, selecciona un perfil y comenta sus promedios.
Conserva el comando local `python -m streamlit run dashboard/app.py` como respaldo.
La entrega debe incluir además la demostración desde la URL desplegada.

## Preguntas que debes poder responder

1. **¿Por qué no sumar todas las ventas?** Los importes usan monedas distintas;
   sumar EUR, USD e INR sin conversión carece de una unidad monetaria común.
2. **¿Por qué retirar puntos y datos de entrega?** En los datos se relacionaban con
   el resultado final y podían revelar la devolución o no existir al predecir.
3. **¿La puntuación 0,50 significa 50 % de probabilidad real?** No se ha comprobado
   calibración; se muestra como puntuación del clasificador balanceado.
4. **¿Por qué la exactitud no basta?** Hay desbalance; un modelo puede acertar la
   clase frecuente sin detectar devoluciones. También hay que revisar precisión,
   recall, F1, AUC y Average Precision.
5. **¿Cómo se eligió el umbral?** Maximizando F1 en validación; prueba se reservó
   para la evaluación final. Se mantuvo 0,491138 en el dashboard.
6. **¿Los grupos representan tipos permanentes de cliente?** Describen pedidos;
   un cliente puede generar pedidos que caigan en varios grupos.
7. **¿Por qué K=4?** Por la combinación del análisis del codo y la interpretación;
   debe reconocerse que la silueta no necesariamente tiene su máximo en K=4.
8. **¿PCA cambia los clústeres?** No: representa en 2D los datos transformados;
   las etiquetas provienen de K-Means en sus tres características originales.
9. **¿Hay impacto de negocio demostrado?** No. Se proponen pilotos con datos reales.
10. **¿Cómo generaliza a clientes nuevos?** La partición por pedido no resuelve esa
    pregunta; falta evaluación separada por cliente y por tiempo.
