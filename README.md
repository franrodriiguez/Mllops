# Práctica MLOps / LLMOps

## Estructura del proyecto

### Notebook MLflow

El notebook principal contiene toda la parte correspondiente a MLflow:

- Carga y exploración del dataset.
- Entrenamiento del modelo KNN.
- Registro de experimentos con MLflow.
- Comparación de métricas (Accuracy, Precision, Recall y F1-Score).
- Selección del mejor modelo.

---

### Notebook FastAPI

Se incluye un segundo notebook con las capturas solicitadas para la parte de FastAPI:

- Captura de la documentación Swagger (`/docs`).
- Captura de cada endpoint con ejemplos de ejecución.
- Captura de las llamadas HTTP realizadas a cada endpoint.

---

### Carpeta `app`

La carpeta `app` contiene todo el código utilizado para desplegar la API con FastAPI.

Los endpoints implementados son:

- `/text-stats`
- `/text-cleaner`
- `/keyword-count`
- `/token_classifier`
- `/fill-mask`

---

### Carpeta `img`

La carpeta `img` contiene las capturas de pantalla utilizadas en el notebook de FastAPI.

Estas capturas incluyen:

- Documentación Swagger (`/docs`).
- Respuestas de cada endpoint desde Swagger.
- Llamadas HTTP realizadas a los endpoints.
- Evidencias de funcionamiento de la API.

---

## Tecnologías utilizadas

- Python
- FastAPI
- MLflow
- Scikit-Learn
- Transformers (Hugging Face)
- NLTK
- PyTorch

---

## Autor

Francisco Rodríguez-Córdoba Lucena
