# Dashboard de pedidos de comercio electrónico

Proyecto integrador de Ciencia de Datos e Inteligencia Artificial.
Jennifer Juliana Bayona Rojas · Universidad de La Salle · Fase 4.

## Qué contiene

Dashboard en Streamlit con filtros por fecha, moneda, país, canal, segmento, estado,
edad, unidades e importe. Incluye indicadores, predicción con los modelos de Fase 3,
mapa interactivo PCA de K-Means y resultados técnicos.

Los datos son transacciones simuladas. La regresión y el clustering están definidos
para USD. La clasificación es exploratoria: AUC-ROC de prueba 0,5605 y precisión 8,38 %.
La puntuación del clasificador no se presenta como una probabilidad calibrada.

## Empezar por el notebook

Abre `notebooks/Fase_4_PROYECTO.ipynb` en Jupyter o Google Colab y sigue las celdas en
orden. El notebook explica cada paso y contiene el código que genera esta aplicación.
Puedes usar esta carpeta ya preparada o trabajar desde `artefactos_fase3.zip`.

## Ejecutar el dashboard en tu computador

Usa Python 3.12. Desde esta carpeta:

```bash
python -m venv .venv
```

Activa el entorno. En Windows: `.venv\Scripts\activate`. En macOS/Linux:
`source .venv/bin/activate`. Después:

```bash
python -m pip install -r requirements.txt
python src/verificar.py
python -m streamlit run dashboard/app.py
```

Abre la dirección local que indica Streamlit. Para editar/ejecutar notebooks también
puedes instalar `python -m pip install -r requirements_notebook.txt`.

## Publicar con GitHub y Streamlit Community Cloud

1. Crea en https://github.com/new un repositorio público llamado, por ejemplo,
   `proyecto-ciencia-datos-ecommerce`. La guía de la asignatura pide que sea público.
2. Descomprime la entrega y sube **los archivos descomprimidos** conservando las
   carpetas. `requirements.txt` debe quedar en la raíz y el script en `dashboard/app.py`.
   Usa **Add file → Upload files** o Git. Subir solamente el ZIP no permite ejecutar la app.
3. Comprueba que se ven `data/`, `models/`, `src/`, `dashboard/`, `notebooks/` y `reports/`.
4. Entra en https://share.streamlit.io/ y conecta la cuenta de GitHub que tiene el repo.
5. Selecciona **Create app** y la opción para una app existente. Indica:

   | Campo | Valor |
   |---|---|
   | Repository | tu-usuario/proyecto-ciencia-datos-ecommerce |
   | Branch | main (o la rama en la que subiste los archivos) |
   | Main file path | dashboard/app.py |
   | Advanced settings → Python version | 3.12 |

6. Pulsa **Deploy**. La instalación puede tardar unos minutos. Revisa los logs si falla.
7. Abre la URL real que te entregue Streamlit. Prueba filtros, una predicción en USD,
   una clasificación con otra moneda y el mapa de clústeres.
8. Comprueba el acceso en una ventana privada. Si solicita iniciar sesión, revisa la
   configuración de acceso para que el docente pueda abrirlo como esperas.
9. Añade la URL real al README, al notebook y al informe; registra también la URL del
   repositorio y el commit presentado. No se ha publicado una URL desde esta entrega.

GitHub almacena el código y Streamlit Community Cloud ejecuta Python. GitHub Pages
publica sitios estáticos y no ejecuta este servidor Streamlit.

Para actualizar la app, sube los cambios al mismo repositorio. Conserva las versiones
fijadas para evitar cambios inesperados al cargar los modelos.

## Archivos principales

| Ruta | Propósito |
|---|---|
| data/dataset_procesado.csv | Los 138.116 pedidos de la Fase 3 |
| data/segmentos_kmeans.csv | Segmentos de los 82.600 pedidos en USD |
| data/base_depurada_fase1.csv | Copia de los datos de entrada de Fase 2 |
| models/ | Modelos y transformadores exportados en Fase 3 |
| notebooks/ | Notebooks de las cuatro fases |
| src/datos.py | Unión validada, filtros, indicadores y perfiles |
| src/prediccion.py | Inferencia, umbral original y proyección PCA |
| src/verificar.py | Integridad y reproducción de métricas de prueba |
| dashboard/app.py | Aplicación web |
| reports/ | Métricas, borrador del informe y guion de sustentación |
| requirements.txt | Dependencias fijadas del dashboard y la inferencia |

## Reproducibilidad y alcance

La inferencia se comprobó con Python 3.12, scikit-learn 1.6.1 y las versiones de
`requirements.txt`. Esta es la misma versión de scikit-learn de los modelos guardados.
El script `src/verificar.py` reconstruye las divisiones originales 70/15/15 usando
`random_state=42` y reproduce las métricas de prueba guardadas, sin volver a entrenar.
También comprueba que los 82.600 clústeres coincidan con los exportados.

`reports/requirements_fase3_original.txt` conserva el entorno registrado en el notebook
original, que difiere del entorno probado de inferencia. No se usa para desplegar.
Los modelos de DBSCAN y MLP se conservan como evidencia; el dashboard sirve los
Random Forest seleccionados y K-Means. No necesita instalar PyTorch para predecir.

Los notebooks de Fase 1–3 se conservaron como fueron adjuntados. La repetición de todo
el entrenamiento depende de los archivos de origen y de esos entornos originales.
En especial, el notebook original de Fase 2 descarga `base_datos.csv`; puedes usar la
copia local `data/base_depurada_fase1.csv` en esa celda. El notebook original de Fase 3
espera `fase2_para_fase3.joblib`, que se obtiene al ejecutar Fase 2. El despliegue y la
Fase 4 funcionan directamente con los archivos incluidos y no descargan esos datos.

Las métricas de modelo corresponden a las particiones originales. Los filtros cambian
indicadores descriptivos, no las métricas de prueba. No se ajustan modelos en la app.
La separación por pedido permite clientes compartidos entre particiones; faltan
evaluaciones temporales y por cliente. Los perfiles son de pedidos y no identidades
permanentes de clientes. Las hipótesis comerciales requieren un piloto real.

## Procedencia

Los archivos recibidos para el proyecto se registran con SHA-256 en
`reports/procedencia_archivos.json`. Según la ficha de Fase 1: E-Commerce Sales
Analytics Dataset, Shair Khan (datascikhan), transacciones simuladas y licencia CC0
declarada allí. Fuente: https://www.kaggle.com/datasets/datascikhan/e-commerce-sales-and-customer-analytics
La información de licencia debe revisarse en la fuente al preparar la entrega pública.

## Documentación oficial de publicación

- https://docs.streamlit.io/deploy/streamlit-community-cloud
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site

## Pendientes de entrega

- Publicar la aplicación y registrar sus URL reales.
- Ejecutar una demostración desde la URL publicada y guardar la evidencia.
- Revisar y convertir el borrador técnico a PDF con los enlaces finales.
- Descargar el notebook ejecutado de Colab y subir esa versión a `notebooks/`.
- Ensayar la exposición de 12 minutos y las preguntas.
