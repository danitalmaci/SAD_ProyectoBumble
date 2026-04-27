# Módulo de Clustering y Topic Modeling

Este módulo forma parte del proyecto de análisis de reseñas de la App **Bumble**. Su objetivo principal es descubrir los temas latentes (*topics*) en las opiniones de los usuarios, segmentándolos por sentimiento (Positivo, Negativo y Neutro) para extraer *insights* accionables de negocio.

## Descripción General
El script `clustering.py` implementa un pipeline completo de Procesamiento de Lenguaje Natural (NLP) que permite:
1.  **Clasificación interna:** Divide las reseñas según su puntuación (*Score*) en categorías de sentimiento.
2.  **Preprocesado:** Limpieza, tokenización, eliminación de *stopwords* y *stemming*.
3.  **Topic Modeling (LDA):** Uso de *Latent Dirichlet Allocation* (vía Gensim) para descubrir temáticas probabilísticas.
4.  **Análisis de Optimización:** Generación automática del "Gráfico del Codo" para validar el número de clústeres.

---

## Configuración (JSON)

### Sección: `preprocessing`
* `text_column`: Nombre de la columna en el CSV que contiene el texto (ej. "content").
* `score_column`: Nombre de la columna que contiene la valoración (ej. "score").
* `language`: Idioma para la carga de *stopwords* de NLTK (ej. "english").
* `extra_stopwords`: Lista personalizada de palabras a ignorar (ruido de dominio como "app", "bumble", etc.).

### Sección: `clustering`
* `algorithm`: Define el algoritmo de clustering a utilizar (`lda` o `kmeans`).
* `k_optimo`: Número de clusters que el modelo intentará encontrar.
* `max_df`: Filtra palabras que aparecen en más del X% de los documentos (elimina ruido común).
* `min_df`: Filtra palabras que aparecen en menos de X documentos (elimina erratas y palabras raras).
* `lda_passes`: Número de iteraciones del modelo sobre el corpus (mayor valor = mayor precisión).
* `lda_alpha`: Controla la distribución de tópicos (se recomienda `"asymmetric"` para casos reales).
* `generar_codo`: Booleano (`true`/`false`) para activar la creación de gráficos.

---

## Estructura del Código
* `load_data()`: Carga el archivo CSV utilizando rutas absolutas para evitar errores de directorio.
* `procesar_texto()`: Aplica la lógica de limpieza (minúsculas, limpieza de signos, eliminación de números y *stemming*).
* `ejecutar_lda_gensim()`: Motor principal. Crea el diccionario y el corpus de Bag-of-Words para extraer las palabras clave por tópico.
* `ejecutar_kmeans()`: Motor secundario. Utiliza vectorización TF-IDF para agrupar documentos por similitud de contenido.
* `metodo_del_codo()`: Genera visualizaciones en la carpeta `/output` para justificar la elección de `K`.

---

## Ejecución

```bash
python clustering/clustering.py -j clustering/configuration.json -f datos/Bumble_limpio.csv