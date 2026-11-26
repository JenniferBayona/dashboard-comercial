# Estimación de ventas, devoluciones y perfiles de pedidos de comercio electrónico

**Autora:** Jennifer Juliana Bayona Rojas.  
**Programa:** Maestría en Inteligencia Artificial, Universidad de La Salle.  
**Estado:** borrador técnico basado en los archivos de Fase 1–3 y la implementación local de Fase 4.
La publicación y la demostración desde una URL pública están pendientes.

## Resumen

Se construyó una solución analítica para estimar la venta neta de pedidos en USD,
explorar la clasificación de devoluciones y describir perfiles de pedidos. El conjunto
procesado contiene 138.116 pedidos; 82.600 están expresados en USD y 123.021 tienen
estado Completed o Returned. Se reutilizaron los pipelines y resultados exportados
en la Fase 3 para desarrollar una aplicación Streamlit con filtros, indicadores,
predicción interactiva y segmentación visual. Random Forest de regresión obtuvo en
prueba MAE de 608,75 USD, RMSE de 885,12 USD y R² de 0,3824. El clasificador de
devoluciones alcanzó AUC-ROC de 0,5605 y precisión de 8,38 %, por lo que su alcance
es exploratorio. K-Means con cuatro grupos se utilizó para visualizar perfiles de
pedidos en USD. Los datos son simulados y no permiten afirmar impacto comercial real.

## 1. Introducción

La información de pedidos permite explorar cómo se relacionan las características
de compra con el importe registrado y las devoluciones. La pregunta del proyecto
es cómo utilizar esas características para estimar la venta neta en USD, identificar
señales asociadas con devoluciones y describir perfiles de pedidos.

El objetivo es integrar preparación, evaluación e interacción en una aplicación
que permita consultar resultados y reconocer sus límites. La unidad principal es
un pedido; un mismo cliente puede aparecer en varias transacciones.

## 2. Metodología

### 2.1 Datos y preparación

Se utilizó la tabla procesada del E-Commerce Sales Analytics Dataset de Kaggle.
Según la ficha de Fase 1, los datos representan transacciones simuladas entre
2021-01-01 y 2025-12-31. La base depurada tiene 18 columnas y 138.116 pedidos.
La tabla exportada de Fase 3 incorpora variables derivadas y `indice_origen`.
La regresión y el clustering se restringen a USD para conservar una unidad monetaria
comparable. No se suman monedas distintas ni se simulan conversiones de divisas.

En Fase 2 se revisó el mecanismo de ausencia de entrega y reseñas, que se relaciona
con el estado del pedido. Esos campos y los puntos de fidelización se excluyeron de
los predictores finales por su disponibilidad y posible fuga de información.
Se crearon mes del pedido, indicador de fin de semana y grupo de edad. La codificación
y el preprocesamiento supervisados se integraron en pipelines ajustados con entrenamiento.

### 2.2 Modelos y evaluación

Se conservaron divisiones de 70 % entrenamiento, 15 % validación y 15 % prueba,
con semilla 42; la clasificación utiliza estratificación. En Fase 3 se compararon
Ridge, Random Forest y una MLP para regresión, y Random Forest y SVM para clasificación.
Los Random Forest seleccionados se optimizaron con validación cruzada; el umbral
clasificador 0,491138 se seleccionó en validación y se mantuvo fijo en prueba.

Para clustering se transformaron cantidad e importe con log1p y se estandarizaron
junto con edad. Se compararon K-Means y DBSCAN. Se eligió K=4 para el mapa principal,
considerando el codo y la interpretación, aunque la silueta favorecía otra configuración
según la narrativa de Fase 3. PCA se utilizó únicamente para representar los grupos.

### 2.3 Implementación de Fase 4

Se implementaron cuatro vistas: resumen comercial, simulador, segmentos y resultados.
Los filtros incluyen fecha, moneda, país, canal, segmento, estado, edad, unidades
e importe. La unión de clústeres se valida mediante `indice_origen` y las variables
originales. Se reutilizan los pipelines, el umbral, el escalador, K-Means y PCA,
sin ajustes durante la interacción.

La venta y el ticket corresponden a pedidos Completed dentro de cada filtro. La tasa
de devolución es Returned / (Completed + Returned) dentro del conjunto seleccionado.
El simulador de ventas se restringe a USD. La salida de clasificación se denomina
puntuación, porque no se ha verificado su calibración. El mapa usa hasta 5.000 puntos
para facilitar la interacción y calcula perfiles con todas las filas seleccionadas.

## 3. Resultados

### 3.1 Regresión en el conjunto de prueba

| Métrica | Resultado |
|---|---:|
| Pedidos de prueba | 12.390 |
| MAE | 608,7492 USD |
| RMSE | 885,1159 USD |
| R² | 0,3824 |

La diferencia de RMSE frente a validación fue de aproximadamente 4,71 USD.
El error medio absoluto aporta una referencia de magnitud; no constituye un
intervalo de confianza para un pedido individual.

### 3.2 Clasificación en el conjunto de prueba

| Métrica | Resultado |
|---|---:|
| Pedidos de prueba | 18.454 |
| Accuracy | 0,3300 |
| Precisión | 0,0838 |
| Sensibilidad | 0,7761 |
| F1 | 0,1513 |
| AUC-ROC | 0,5605 |
| Average Precision | 0,0999 |

Con el umbral seleccionado se detectaron 1.102 de las 1.420 devoluciones reales,
pero 12.046 de las 13.148 alertas fueron falsos positivos, según Fase 3.
El desempeño limita el uso del clasificador como apoyo exploratorio.

### 3.3 Clustering

| Modelo | Configuración | Registros de ajuste | Silueta exportada | Ruido |
|---|---|---:|---:|---:|
| K-Means | K=4 | 82.600 | 0,3028 | 0 % |
| DBSCAN | eps=0,481; min_samples=20 | 8.000 | 0,2081 | 0,85 % |

Las siluetas se compararon sobre las mismas 7.932 observaciones sin ruido de DBSCAN.
Los perfiles de K-Means combinan diferencias de edad, unidades y valor. Las dos
componentes PCA conservan aproximadamente el 89,85 % de la variabilidad transformada.
Los nombres jóvenes/mayores y alto/menor valor son etiquetas descriptivas relativas
a los promedios de los grupos, no límites demográficos o monetarios oficiales.

### 3.4 Validación de la aplicación

El script `src/verificar.py` reproduce las métricas de prueba de los modelos guardados
y contrasta las 82.600 etiquetas K-Means con las exportadas. La versión de inferencia
se fija en `requirements.txt` y usa scikit-learn 1.6.1. La publicación pública se debe
comprobar después del despliegue; una verificación local no prueba acceso remoto.

## 4. Discusión

La regresión captura parte de la variabilidad de ventas, pero conserva errores
considerables para decisiones por pedido. El clasificador tiene una mejora reducida
en discriminación y una elevada tasa de falsas alertas. Un mayor recall no basta
para justificar decisiones operativas sin evaluar sus costos y su precisión.

Los segmentos permiten formular hipótesis para futuras acciones comerciales, como
paquetes complementarios o beneficios para la compra siguiente. No se ha ejecutado
un experimento ni medido incremento de ventas, recompra o rentabilidad. Para evaluar
esas acciones se propone asignar tratamientos por cliente y medir resultados a 30 días.

La partición por pedido permite clientes compartidos entre conjuntos. La evaluación
no mide directamente desempeño con clientes nuevos ni predicción hacia el futuro.
La naturaleza simulada de los datos restringe la generalización a una operación real.

## 5. Conclusiones

Se integraron los resultados del proyecto en una aplicación interactiva que permite
filtrar pedidos, consultar indicadores, simular entradas y visualizar cuatro perfiles
en USD. Los modelos se sirven con sus transformaciones originales y sin reajuste.
Random Forest de regresión ofrece una estimación con precisión moderada; el
clasificador de devoluciones requiere mejoras antes de considerar un uso operativo.
K-Means facilita la lectura de perfiles descriptivos y el planteamiento de pilotos.

## 6. Trabajo futuro

Evaluar particiones por cliente y por tiempo; incorporar características disponibles
antes de la devolución; revisar calibración y costos de falsas alertas; estudiar errores
por subgrupo; validar estabilidad de clústeres y probar la solución con datos reales.

## 7. Publicación y evidencias por completar

- URL del repositorio público: **PENDIENTE**.
- URL real del dashboard: **PENDIENTE**.
- Commit de la versión presentada: **PENDIENTE**.
- Fecha y evidencia de prueba desde la URL pública: **PENDIENTE**.
- Figuras del dashboard publicado y revisión final de referencias: **PENDIENTE**.

## Referencias

1. Guía técnica del Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial,
   Universidad de La Salle, periodo octubre–noviembre de 2026, documento proporcionado.
2. Bayona Rojas, J. J. Notebooks de Fase 1, Fase 2 y Fase 3 y artefactos proporcionados.
3. Dataset: https://www.kaggle.com/datasets/datascikhan/e-commerce-sales-and-customer-analytics
4. Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud
5. Guía de despliegue: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
