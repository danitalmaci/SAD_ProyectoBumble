# Clustering

Este módulo es la parte de aprendizaje no supervisado del proyecto. Básicamente, coge los comentarios de la app (separando previamente los positivos de los negativos) y usa K-Means junto con TF-IDF para agruparlos. 

El objetivo no es predecir nada, sino entender **de qué habla la gente**. Por ejemplo, si ponen nota baja, saber si es por la interfaz, por el precio o por perfiles falsos.

## Cómo ejecutarlo
```bash
python clustering.py -j configuration.json -f archivo.csv