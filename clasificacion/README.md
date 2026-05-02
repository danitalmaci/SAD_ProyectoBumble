# SAD: Plantilla de prototipos de clasificación

## Miembros
* **June Castro**
* **Daniel Talmaci**

---

## Objetivo y Descripción del proyecto

Este repositorio contiene un pipeline automatizado de Machine Learning diseñado para optimizar el desarrollo, entrenamiento y evaluación de modelos de clasificación. El sistema permite identificar las necesidades específicas de cada algoritmo y aplicar las estrategias de preprocesamiento y balanceo más adecuadas en cada caso, permitiendo seleccionar el mejor modelo según métricas de rendimiento definidas.
Los algoritmos implementados son los siguientes:
    - KNN
    - Decision Trees
    - Random Forest
    - Naïve Bayes

---

## Requisitos de Software

Para asegurar el correcto funcionamiento del pipeline, es necesario utilizar la siguiente versión de Python y contar con las dependencias instaladas:

* **Versión de Python:** `3.13.12`
* **Instalación de dependencias:**
  ```bash
  pip install -r requirements.txt
  
---

## Contenido del Proyecto

* **`train.py`**: Script principal para el entrenamiento y optimización de modelos.
* **`test.py`**: Script para la evaluación de modelos y generación de predicciones.
* **`config_train.json`**: Plantilla de configuración para la fase de entrenamiento.
* **`config_test.json`**: Plantilla de configuración para la fase de testeo.
* **`requirements.txt`**: Lista de librerías de Python necesarias (pandas, scikit-learn, etc.).

---

## Guía de uso

El proyecto se divide en dos fases principales: entrenamiento y testeo. 
Ambas se gestionan mediante scripts de terminal que aceptan archivos de configuración en formato JSON.

1. Entrenamiento de Modelos (train.py) + config_train.json

* **Comando de ejecución:**
  ```bash
  python train.py -j config_train.json 

2. Testeo y Predicción (test.py) + config_test.json

* **Comando de ejecución:**
  ```bash
  python train.py -j config_test.json 
