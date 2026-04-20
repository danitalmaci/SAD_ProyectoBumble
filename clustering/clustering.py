# -*- coding: utf-8 -*-  

import sys
import argparse
import pandas as pd
import string
import os
import matplotlib.pyplot as plt
from colorama import Fore

# Sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Nltk 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ----------------- Funciones auxiliares -----------------  # Separador de funciones auxiliares.

def parse_args(): 
    parse = argparse.ArgumentParser(description="Script de Clustering para Análisis de Sentimientos.")
    parse.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv con los datos", required=True)
    args = parse.parse_args()

    import json
    try:  
        with open(args.json, 'r') as json_file:
            config = json.load(json_file)
        for key, value in config.items():
            setattr(args, key, value)
    except FileNotFoundError: 
        print(f"Error: No se encontró el archivo {args.json}")  
        sys.exit(1)  

    return args 

def load_data(file):  
    try: 
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN + f"Datos cargados con éxito: {len(data)} filas." + Fore.RESET)
        return data
    except Exception as e:  
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e) 
        sys.exit(1) 

# -------------- Funciones de Preprocesado -------------- 

def procesar_texto(texto, stop_words, stemmer):
    tokens = word_tokenize(str(texto).lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha()] # Solo letras
    return " ".join(tokens)

def limpiar_y_vectorizar(data, columna_texto):
    print("\n- Simplificando y vectorizando el texto (TF-IDF)...")
    language = args.preprocessing.get("language", "spanish")
    stop_words = set(stopwords.words(language))
    
    # Se pueden añadir STOP WORDS
    palabras_extra = args.preprocessing.get("extra_stopwords", [])
    stop_words.update(palabras_extra)

    stemmer = PorterStemmer()

    # Aplicar limpieza
    data['texto_limpio'] = data[columna_texto].apply(lambda x: procesar_texto(x, stop_words, stemmer))

    # Vectorización TF-IDF
    max_df = args.clustering.get("max_df", 0.95)
    min_df = args.clustering.get("min_df", 2)
    
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    X_tfidf = vectorizer.fit_transform(data['texto_limpio'])
    
    print(Fore.GREEN + f"Texto vectorizado. Matriz final: {X_tfidf.shape}" + Fore.RESET)
    return X_tfidf, vectorizer

# ----------------- Funciones de Clustering ----------------- 

def metodo_del_codo(X_tfidf, sentimiento):
    print(Fore.CYAN + f"\n- Ejecutando Método del Codo para críticas {sentimiento}..." + Fore.RESET)
    rango_k = range(2, args.clustering.get("max_k", 10) + 1)
    inercias = []

    for k in rango_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_tfidf)
        inercias.append(kmeans.inertia_)

    # Crear gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(rango_k, inercias, marker='o', linestyle='-', color='b')
    plt.title(f'Método del Codo - Críticas {sentimiento.capitalize()}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia')
    plt.grid(True)
    
    # Guardar gráfico
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_grafico = os.path.join(directorio_actual, 'output', f'codo_{sentimiento}.png')
    plt.savefig(ruta_grafico)
    print(Fore.GREEN + f"Gráfico guardado en {ruta_grafico}" + Fore.RESET)
    plt.close()

def ejecutar_kmeans(X_tfidf, vectorizer, data, sentimiento):
    # Cogemos el K óptimo definido en el JSON
    k_optimo = args.clustering.get("k_optimo", 4)
    print(Fore.CYAN + f"\n- Ajustando K-Means final con K={k_optimo} para {sentimiento}..." + Fore.RESET)

    kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    kmeans.fit(X_tfidf)

    # Asignar cluster a cada comentario
    data['Cluster'] = kmeans.labels_

    # Extraer palabras top
    terminos = vectorizer.get_feature_names_out()
    centroides = kmeans.cluster_centers_
    num_palabras = args.clustering.get("top_words", 10)

    print(Fore.MAGENTA + f"\nTop {num_palabras} palabras por Cluster ({sentimiento}):" + Fore.RESET)
    
    # Para guardar las palabras en el df de Tableau
    palabras_por_cluster = {}
    
    for i in range(k_optimo):
        indices_top = centroides[i].argsort()[:-num_palabras - 1:-1]
        palabras_cluster = [terminos[ind] for ind in indices_top]
        palabras_por_cluster[i] = ", ".join(palabras_cluster)
        print(f"  > Cluster {i}: {palabras_por_cluster[i]}")

    # Añadir las palabras clave al dataframe para Tableau
    data['Palabras_Clave'] = data['Cluster'].map(palabras_por_cluster)

    # Guardar datos Tableau
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(directorio_actual, 'output', f'tableau_datos_{sentimiento}.csv')
    data.to_csv(ruta_csv, index=False)
    print(Fore.GREEN + f"Datos listos para Tableau guardados en {ruta_csv}" + Fore.RESET)

# ======================= MAIN ======================= 

if __name__ == "__main__":  
    print("=== Script de Clustering ===")
    args = parse_args()

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    carpeta_output = os.path.join(directorio_actual, 'output')

    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)

    # Cargar datos
    data = load_data(args.file)
    columna_texto = args.preprocessing["text_column"]
    columna_score = "score"

    print(Fore.CYAN + "\n- Pasando de score a sentimientos..." + Fore.RESET)
    
    def clasificar_estrellas(estrellas):
        try:
            estrellas = float(estrellas)
            if estrellas <= 2:
                return 'negativa'
            elif estrellas == 3:
                return 'neutra'
            else:
                return 'positiva'
        except:
            return 'desconocido' # Por si hay alguna fila vacía o asi

    # nueva columna para separar
    data['sentimiento_real'] = data[columna_score].apply(clasificar_estrellas)
    
    print(data['sentimiento_real'].value_counts())

    # ----------------- Negativas ----------------- 
    print(Fore.YELLOW + "\n--- ANALIZANDO CRÍTICAS NEGATIVAS ---" + Fore.RESET)
    data_negativa = data[data['sentimiento_real'] == 'negativa'].copy()
    
    if len(data_negativa) > 0:
        X_tfidf_neg, vectorizer_neg = limpiar_y_vectorizar(data_negativa, columna_texto)
        metodo_del_codo(X_tfidf_neg, "negativas")
        ejecutar_kmeans(X_tfidf_neg, vectorizer_neg, data_negativa, "negativas")
    else:
        print("No se encontraron críticas negativas.")

    # ----------------- Positivas ----------------- 
    print(Fore.YELLOW + "\n--- ANALIZANDO CRÍTICAS POSITIVAS ---" + Fore.RESET)
    data_positiva = data[data['sentimiento_real'] == 'positiva'].copy()
    
    if len(data_positiva) > 0:
        X_tfidf_pos, vectorizer_pos = limpiar_y_vectorizar(data_positiva, columna_texto)
        metodo_del_codo(X_tfidf_pos, "positivas") 
        ejecutar_kmeans(X_tfidf_pos, vectorizer_pos, data_positiva, "positivas")
    else:
        print("No se encontraron críticas positivas.")