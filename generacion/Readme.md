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
| `--csv`       | str     | Si-predict    | Ruta al archivo CSV                                         |
| `--target`    | str     | Si-predict    | Columna que contiene el texto de opinión                    |
| `--model`     | str     | No            | Modelo de Ollama (por defecto: `llama3:8b-text-q2_K`)       |
| `--shot`      | str     | No            | Tipo de prompting: `0`, `1` o `few` (por defecto: `0`)      |
| `--sentiment` | str/int | No-predict    | Columna numérica 1–5                                        |
| `--samples`   | int     | No            | Número de muestras a procesar o generar (por defecto: `10`) |
| `--score`     | int     | Si-oversample | Score de la opinión a generar                               |
| `--prompt`    | str     | Si-oversample | Archivo .txt con el prompt                                  |

Modelos a recomendados: llama3:8b-text-q2_K / gemma2:2b-text-q4_K_S

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
  --target content \
  --sentiment score \
  --samples 10 \
  --shot 0
```

### Salida

* `predictions.csv` → datos originales + predicción
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
  --score 5 \
  --samples 5
```

Esto:

* Genera 5 comentarios **positivos**
* Guarda el resultado en `oversampled.csv`

---

## Archivos de salida

| Archivo           | Descripción                            |
| ----------------- | -------------------------------------- |
| `predictions.csv` | Resultados del modo `predict`          |
| `oversampled.csv` | Dataset con datos sintéticos           |

---

## Requisitos

* Python 3.9+
* Ollama instalado
* Librerías necesarias:

```bash
pip install pandas scikit-learn langchain langchain-ollama
```