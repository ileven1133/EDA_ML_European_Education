# EDA_ML_European_Education
Trabajo de final de bootcamp consistente en una EDA y un entreno de modelos de ML
# Análisis de Datos Socioeconómicos y Predicción del Nivel de Éxito y Educación General

Este repositorio contiene el código y los resultados de un análisis de datos socioeconómicos con el objetivo de predecir dos variables clave: el "nivel de éxito" (categorizado a partir de porcentajes de educación y empleo) y el "porcentaje de educación general".

## Contenido del Repositorio

* `data_final_classification.csv`: Archivo CSV que contiene los datos limpios y preprocesados para la tarea de clasificación.
* `data_cleaned_regression.csv`: Archivo CSV que contiene los datos limpios y preprocesados para la tarea de regresión.
* `Modelado_Educacion_Europa_Regresion.ipynb` o `Modelado_Educacion_Europa_Clasificacion.ipynb`: Notebook de Jupyter o script de Python que contiene el código para la carga de datos, preprocesamiento, entrenamiento y evaluación de los modelos de clasificación y regresión. (Reemplaza con el nombre real de tu archivo principal).
* `README.md`: Este archivo, que proporciona una descripción general del proyecto.
* `[Otros archivos o directorios relevantes]`: Cualquier otro archivo, como visualizaciones guardadas o resultados detallados.

## Descripción del Proyecto

Este proyecto aborda la predicción de dos aspectos importantes del desarrollo socioeconómico:

1.  **Nivel de Éxito (Clasificación):** Se creó una variable categórica de "nivel de éxito" basada en la combinación de los percentiles de "percentage\_education\_general" y "percentage\_by\_employment\_status". Se utilizaron modelos de clasificación (Regresión Logística y Random Forest) para predecir si un individuo o grupo se clasificaría como de nivel de éxito "Alto", "Medio" o "Bajo".

2.  **Porcentaje de Educación General (Regresión):** Se utilizaron modelos de regresión (Regresión Lineal y Random Forest Regressor) para predecir el valor continuo de "percentage\_education\_general" basándose en otras variables socioeconómicas.

El objetivo principal fue identificar los factores clave que influyen en estas variables y construir modelos predictivos precisos.

## Metodología

1.  **Carga y Limpieza de Datos:** Se cargaron y limpiaron los datos relevantes para cada tarea (clasificación y regresión).
2.  **Preprocesamiento de Datos:** Se aplicaron técnicas de preprocesamiento como imputación de valores faltantes, escalado de características numéricas y codificación one-hot para características categóricas. Se utilizó `sklearn.compose.ColumnTransformer` para aplicar diferentes transformaciones a diferentes tipos de columnas.
3.  **División de Datos:** Los datos se dividieron en conjuntos de entrenamiento, validación y prueba para el ajuste de hiperparámetros y la evaluación final de los modelos. Se aplicó muestreo estratificado para la tarea de clasificación.
4.  **Modelado:**
    * **Clasificación:** Se entrenaron y evaluaron modelos de Regresión Logística y Random Forest Classifier, con y sin reducción de dimensionalidad mediante PCA. Se utilizaron métricas como accuracy, precisión, recall, F1-score, matriz de confusión y curvas ROC/AUC para evaluar el rendimiento.
    * **Regresión:** Se entrenaron y evaluaron modelos de Regresión Lineal y Random Forest Regressor. Se utilizaron métricas como Mean Squared Error (MSE), Mean Absolute Error (MAE) y R-squared (R2) para evaluar el rendimiento.
5.  **Ajuste de Hiperparámetros:** Se utilizó `sklearn.model_selection.GridSearchCV` para encontrar los mejores hiperparámetros para cada modelo utilizando un conjunto de validación.
6.  **Evaluación:** Se evaluó el rendimiento de los modelos con los mejores hiperparámetros en un conjunto de prueba independiente.
7.  **Análisis de Importancia de Características:** Se analizó la importancia de las características para los modelos entrenados (coeficientes para Regresión Lineal e importancia de Gini para Random Forest).
8.  **Visualización:** Se generaron diversas visualizaciones, incluyendo matrices de confusión, curvas ROC (para clasificación), gráficos de dispersión de predicciones vs. valores reales y gráficos de residuos (para regresión), y gráficos de importancia de características.

## Resultados Clave

* **Clasificación:** El **Random Forest sin PCA** demostró ser el modelo más efectivo para predecir el nivel de éxito, alcanzando una accuracy del 89% en el conjunto de prueba. La reducción de dimensionalidad con PCA no mejoró el rendimiento. Los niveles educativos y el grupo de edad fueron predictores importantes.
* **Regresión:** El **Random Forest Regressor** superó a la Regresión Lineal en la predicción del porcentaje de educación general, con un R-squared de 0.97 en el conjunto de prueba. El porcentaje de empleo y los niveles educativos fueron las características más importantes.

## Cómo Ejecutar el Código

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/ileven1133/EDA_ML_European_Education.git](https://github.com/ileven1133/EDA_ML_European_Education.git)
    cd EDA_ML_European_Education
    ```
2.  **Asegurarse de tener las librerías necesarias instaladas:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  **Ejecutar el notebook o script principal:**
    ```bash
    jupyter notebook Modelado_Educacion_Europa_Clasificacion.ipynb
    o
    jupyter notebook Modelado_Educacion_Europa_Regresion.ipynb
    ```
    Asegúrate de que los archivos `.csv` estén en el mismo directorio que el script o notebook, o proporciona las rutas correctas.

## Trabajo Futuro

* Explorar modelos de machine learning más avanzados (e.g., Gradient Boosting Machines, Redes Neuronales).
* Realizar una ingeniería de características más profunda.
* Incorporar datos adicionales relevantes.
* Profundizar en la interpretabilidad de los modelos complejos.
* Realizar una validación cruzada más robusta.
* Considerar el despliegue y la monitorización de los modelos.

## Autor

Ivan Vendrell
ivan.vendrell@gmail.com

