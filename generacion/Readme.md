# Clasificación de Sentimiento y Aumento de Datos con Ollama

## Descripción general

Este script permite:

1. **Clasificar el sentimiento** (Positive, Negative, Neutral) de textos usando un modelo LLM en local con Ollama.
2. **Generar datos sintéticos (oversampling)** creando nuevos comentarios de opinión.

---

## Funcionalidades

* Clasificación de sentimiento con prompting (zero-shot, one-shot, few-shot)
* Mapeo automático de valores numéricos (1–5) a etiquetas de sentimiento
* Generación de textos sintéticos con sentimiento controlado
* Entrada y salida en CSV 

---

## Mapeo de sentimiento

Si se proporciona una columna numérica (1–5), se convierte así:

| Score | Sentimiento |
| ----- | ----------- |
| 1–2   | Negative    |
| 3     | Neutral     |
| 4–5   | Positive    |

---

## Argumentos

| Argumento     | Tipo    | Obligatorio   | Descripción                                                 |
| ------------- | ------- | ------------- | ----------------------------------------------------------- |
| `--mode`      | str     | Si            | Modo de ejecución: `predict` o `oversample`                 |
| `--prompt`    | str     | Si            | Archivo .txt con el prompt                                  |
| `--csv`       | str     | Si-predict    | Ruta al archivo CSV                                         |
| `--target`    | str     | Si-predict    | Columna que contiene el texto de opinión                    |
| `--sentiment` | str/int | Si-predict    | Columna numérica 1–5                                        |
| `--model`     | str     | No            | Modelo de Ollama (por defecto: `llama3:8b-text-q2_K`)       |
| `--samples`   | int     | No            | Número de muestras a procesar o generar (por defecto: `10`) |
| `--score`     | int     | Si-oversample | Score de la opinión a generar                               |


Modelos a recomendados: llama3:8b-text-q2_K / gemma2:2b-text-q4_K_S / granite4:350m-h

---

## Modo `predict`

Este modo:

* Selecciona una muestra aleatoria del dataset
* Predice el sentimiento con el modelo mediante el prompt especificado
* Evalúa resultados comparandolo con las etiquetas reales

### Ejemplo de llamada al script

```bash
python LangchainBubmle.py \
  --mode predict \
  --csv Bumble_corregido.csv \
  --prompt prompt.txt\
  --target content \
  --sentiment score \
  --samples 10 \
```

### Salida

* `predictions_generative.csv` → modelo + prompt + entrada + salida + real
* Nota: Si la salida es `NotExpected`, significa que el modelo no ha respondido con solo [Positive,Neutral,Negative]
* Métricas en consola:

  * Accuracy
  * F1-score
  * Classification report

---

## Modo `oversample`

Este modo:

* Genera **nuevos textos sintéticos**
* Usa un **sentimiento fijo definido por el usuario**
* Añade los nuevos datos al CSV original

### Comportamiento

* Genera exactamente `N` muestras nuevas
* Todas tienen el mismo sentimiento
* Columnas generadas:

  * `content` → texto generado
  * `score` → valor numérico (1–5)

---

### Ejemplo de llamada al script

```bash
python script.py \
  --mode oversample \
  --prompt prompt.txt \
  --score 5 \
  --samples 5
```

Esto:

* Genera 5 comentarios **positivos**
* Guarda el resultado en `nombrePrompt_sentiment.csv`

---

## Archivos de salida

| Archivo                      | Descripción                                |
| ---------------------------- | ------------------------------------------ |
| `predictions.csv`            | Resultados del modo `predict` acumulativos |
| `nombrePrompt_sentiment.csv` | Dataset con datos sintéticos           s    |

---

## Requisitos

* Python 3.9+
* Ollama instalado
* Librerías necesarias:

```bash
pip install pandas scikit-learn langchain langchain-ollama
```