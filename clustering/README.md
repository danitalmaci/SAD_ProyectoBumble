# Plantilla Clustering

## Cómo ejecutarlo

```bash
python clustering.py -j configuration.json -f Bumble_limpio.csv
```

##  Cómo funciona

1. **Clasificación por Sentimiento:** Lee la puntuación de la reseña (`score`) y la divide en tres grupos: Negativas (1-2), Neutras (3) y Positivas (4-5). Los análisis de tópicos se hacen por separado para no mezclar quejas con halagos.
2. **Preprocesamiento NLP:** * Transforma emojis en texto.
   * Elimina signos de puntuación y *stopwords*.
   * Lematiza (recorta las palabras a su raíz léxica).
   * Genera **Bigramas**: Une palabras que suelen aparecer juntas (ej. "customer" y "service" pasan a ser "customer_service").
3. **División Híbrida (Cortos vs. Largos):** Dado que los textos muy cortos (ej. "mala app") arruinan los modelos probabilísticos, el script separa las reseñas según su longitud.
4. **Modelado (LDA + K-Means):** (si se usa el algoritmo hibrido en el JSON)
   * Textos largos: Usa **LDA (Latent Dirichlet Allocation)**. El script evalúa varios números de clústeres (K) y elige automáticamente el que tiene mejor puntuación de "Coherencia".
   * Textos cortos: Usa **K-Means** con vectorización TF-IDF.
5. **Exportación:** Genera gráficos de optimización en la carpeta `/output` y un archivo `clustering_modo_tableau.csv` listo para visualizar.

##  Explicación del JSON

### Preprocesado

* **`text_column`**: El nombre exacto de la columna del CSV que contiene el texto de las reseñas.
* **`score_column`**: La columna con las estrellas o puntuación (necesaria para dividir por sentimientos).
* **`language`**: El idioma principal de los textos para cargar las *stopwords* por defecto de NLTK (ej. "english", "spanish").
* **`extra_stopwords`**: Una lista de palabras específicas de tu contexto que el modelo debe ignorar porque no aportan valor temático (ej. el nombre de la app, palabras obvias como "app", "use", "people").
* **`bigram_min_count`**: Frecuencia mínima. Para que dos palabras se unan en un bigrama, deben aparecer juntas en el dataset al menos este número de veces (ej. 3).
* **`bigram_threshold`**: Nivel de exigencia estadística para formar el bigrama. Un número más bajo une más palabras; un número más alto es más estricto.

### Bloque `clustering`

* **`algorithm`**: Define qué modelo usar. Acepta `"hibrido"`, `"lda"` (fuerza LDA para todo) o `"kmeans"` (fuerza K-Means para todo).
* **`max_df`**: Límite superior de frecuencia. Si está en `0.85`, el modelo ignorará palabras que aparezcan en más del 85% de los documentos (son demasiado comunes).
* **`min_df`**: Límite inferior de frecuencia. Ignorará palabras que aparezcan en menos de `3` documentos en todo el corpus (son demasiado raras o errores tipográficos).
* **`max_k`**: El número máximo de clústeres/tópicos que el LDA intentará buscar. El script probará desde K=3 hasta este número y se quedará con el más coherente.
* **`k_optimo_dict`**: Como el K-Means no elige su K de forma automática en este script, aquí le indicas a mano cuántos clústeres quieres que genere para los textos cortos de cada sentimiento tras haber revisado la gráfica del codo.
* **`top_words`**: Cuántas palabras clave quieres que el script te imprima en la consola y guarde en el CSV para describir cada clúster.
* **`generar_codo`**: `true` o `false`. Si es `true`, generará gráficos de "Método del Codo" y los guardará en la carpeta de salida para que puedas decidir el mejor `K` para K-Means.
* **`lda_passes`**: Número de pasadas (epochs) que el modelo LDA da sobre el corpus completo durante su entrenamiento. Un número alto mejora los resultados a costa de más tiempo de procesamiento.
* **`lda_alpha`**: Densidad de los tópicos. `"asymmetric"` suele funcionar mejor en textos cortos/medianos porque asume que unos tópicos son más comunes que otros.
* **`umbral_palabras_cortas`**: El punto de corte para el modo híbrido. Si está en `5`, cualquier reseña con 5 palabras o menos irá a K-Means; si tiene 6 o más, irá a LDA.
* **`random_state`**: Semilla aleatoria (ej. `42`). Asegura que si ejecutas el código varias veces con los mismos datos, los algoritmos devuelvan exactamente el mismo resultado.