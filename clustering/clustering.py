# -*- coding: utf-8 -*-  
import sys
import argparse
import pandas as pd
import string
import os
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from colorama import Fore

# Sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Nltk 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ----------------- Funciones auxiliares -----------------

def parse_args(): 
    parse = argparse.ArgumentParser(description="Script de Clustering (LDA / K-Means).")
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
        print(Fore.RED + f"Error: No se encontró el archivo {args.json}" + Fore.RESET)  
        sys.exit(1)  

    return args 

def load_data(file):  
    try: 
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_absoluta = os.path.join(directorio_actual, file)
        data = pd.read_csv(ruta_absoluta, encoding='utf-8')
        print(Fore.GREEN + f"Datos cargados con éxito: {len(data)} filas." + Fore.RESET)
        return data
    except Exception as e:  
        print(Fore.RED + "Error al cargar los datos. Comprueba la ruta y el archivo." + Fore.RESET)
        print(e) 
        sys.exit(1) 

def clasificar_estrellas(estrellas):
    try:
        val = float(estrellas)
        if val <= 2: return 'negativa'
        elif val == 3: return 'neutra'
        else: return 'positiva'
    except:
        return 'desconocido'

# ----------------- Preprocesado NLP -----------------

def procesar_texto(texto, stop_words, stemmer):
    tokens = word_tokenize(str(texto).lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha()]
    return " ".join(tokens)

# ----------------- Funciones Clustering -----------------

def ejecutar_lda_gensim(data, columna_texto_limpio, sentimiento, args):
    num_topicos = args.clustering.get("k_optimo", 4)
    min_df = args.clustering.get("min_df", 3)
    max_df = args.clustering.get("max_df", 0.85)
    
    print(Fore.CYAN + f"\n- [LDA] Descubriendo {num_topicos} tópicos para críticas {sentimiento}..." + Fore.RESET)

    textos_tokenizados = [str(texto).split() for texto in data[columna_texto_limpio]]
    id2word = corpora.Dictionary(textos_tokenizados)
    id2word.filter_extremes(no_below=min_df, no_above=max_df)
    corpus = [id2word.doc2bow(texto) for texto in textos_tokenizados]

    lda_model = gensim.models.LdaMulticore(
        corpus=corpus, id2word=id2word, num_topics=num_topicos, random_state=42, passes=10
    )

    print(Fore.MAGENTA + f"\nTop Palabras por Tópico (LDA - {sentimiento}):" + Fore.RESET)
    topicos_crudos = lda_model.show_topics(num_topics=num_topicos, num_words=args.clustering.get("top_words", 10), formatted=False)
    
    palabras_por_topico = {}
    for num_topico, palabras in topicos_crudos:
        lista_palabras = [palabra for palabra, peso in palabras]
        texto_topico = ", ".join(lista_palabras)
        palabras_por_topico[num_topico] = texto_topico
        print(f"  > Tópico {num_topico}: {texto_topico}")

    topicos_dominantes = []
    for bow in corpus:
        if len(bow) == 0:
            topicos_dominantes.append(-1)
            continue
        distribucion = lda_model.get_document_topics(bow)
        topico_principal = sorted(distribucion, key=lambda x: x[1], reverse=True)[0][0]
        topicos_dominantes.append(topico_principal)

    data['Cluster_Dominante'] = topicos_dominantes
    data['Palabras_Clave'] = data['Cluster_Dominante'].map(palabras_por_topico)
    data.loc[data['Cluster_Dominante'] == -1, 'Palabras_Clave'] = "Sin datos"
    
    guardar_resultados(data, f'tableau_lda_{sentimiento}.csv')

def ejecutar_kmeans(data, columna_texto_limpio, sentimiento, args):
    k_optimo = args.clustering.get("k_optimo", 4)
    print(Fore.CYAN + f"\n- [K-MEANS] Vectorizando (TF-IDF) y agrupando en {k_optimo} clusters para {sentimiento}..." + Fore.RESET)

    vectorizer = TfidfVectorizer(max_df=args.clustering.get("max_df", 0.95), min_df=args.clustering.get("min_df", 2))
    X_tfidf = vectorizer.fit_transform(data[columna_texto_limpio])

    kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    kmeans.fit(X_tfidf)
    data['Cluster_Dominante'] = kmeans.labels_

    terminos = vectorizer.get_feature_names_out()
    centroides = kmeans.cluster_centers_
    num_palabras = args.clustering.get("top_words", 10)

    print(Fore.MAGENTA + f"\nTop {num_palabras} palabras por Cluster (K-Means - {sentimiento}):" + Fore.RESET)
    palabras_por_cluster = {}
    
    for i in range(k_optimo):
        indices_top = centroides[i].argsort()[:-num_palabras - 1:-1]
        palabras_cluster = [terminos[ind] for ind in indices_top]
        palabras_por_cluster[i] = ", ".join(palabras_cluster)
        print(f"  > Cluster {i}: {palabras_por_cluster[i]}")

    data['Palabras_Clave'] = data['Cluster_Dominante'].map(palabras_por_cluster)
    guardar_resultados(data, f'tableau_kmeans_{sentimiento}.csv')

def guardar_resultados(data, nombre_archivo):
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(directorio_actual, 'output', nombre_archivo)
    data.to_csv(ruta_csv, index=False)
    print(Fore.GREEN + f"Datos guardados en {ruta_csv}" + Fore.RESET)

def metodo_del_codo(data, columna_texto_limpio, sentimiento, args):
    if not args.clustering.get("generar_codo", False):
        return # config json

    print(Fore.CYAN + f"\n- [MÉTODO DEL CODO] Calculando inercias para {sentimiento}..." + Fore.RESET)
    
    # Vectorizamos solo para evaluar el codo
    vectorizer = TfidfVectorizer(max_df=args.clustering.get("max_df", 0.95), min_df=args.clustering.get("min_df", 2))
    X_tfidf = vectorizer.fit_transform(data[columna_texto_limpio])
    
    rango_k = range(2, args.clustering.get("max_k", 10) + 1)
    inercias = []

    for k in rango_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_tfidf)
        inercias.append(kmeans.inertia_)

    # crear grafico
    plt.figure(figsize=(8, 5))
    plt.plot(rango_k, inercias, marker='o', linestyle='-', color='b')
    plt.title(f'Método del Codo - Críticas {sentimiento.capitalize()}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia')
    plt.grid(True)
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_grafico = os.path.join(directorio_actual, 'output', f'codo_{sentimiento}.png')
    plt.savefig(ruta_grafico)
    print(Fore.GREEN + f"Gráfico del codo guardado en {ruta_grafico}" + Fore.RESET)
    plt.close()

# ----------------- Main -----------------

if __name__ == "__main__":  
    print(Fore.YELLOW + "=== Pipeline de Topic Modeling & Clustering ===" + Fore.RESET)
    args = parse_args()

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    carpeta_output = os.path.join(directorio_actual, 'output')
    os.makedirs(carpeta_output, exist_ok=True)

    data = load_data(args.file)
    columna_texto = args.preprocessing["text_column"]
    columna_score = args.preprocessing.get("score_column", "score")
    
    # Clasificar Sentimientos
    print(Fore.CYAN + "- Clasificando sentimientos..." + Fore.RESET)
    data['sentimiento_analisis'] = data[columna_score].apply(clasificar_estrellas)
    print(Fore.CYAN + "- Aplicando NLTK (Stopwords y Stemming)..." + Fore.RESET)
    language = args.preprocessing.get("language", "spanish")
    
    try:
        stopwords.words(language)
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')

    stop_words = set(stopwords.words(language))
    stop_words.update(args.preprocessing.get("extra_stopwords", []))
    stemmer = PorterStemmer()

    data['texto_limpio'] = data[columna_texto].apply(lambda x: procesar_texto(x, stop_words, stemmer))

    algoritmo = args.clustering.get("algorithm", "lda").lower()
    
    for sentimiento in ['negativa', 'positiva']:
        subset_data = data[data['sentimiento_analisis'] == sentimiento].copy()
        
        if len(subset_data) > 0:
            metodo_del_codo(subset_data, 'texto_limpio', sentimiento, args)
            if algoritmo == "lda":
                ejecutar_lda_gensim(subset_data, 'texto_limpio', sentimiento, args)
            elif algoritmo == "kmeans":
                ejecutar_kmeans(subset_data, 'texto_limpio', sentimiento, args)
            else:
                print(Fore.RED + f"Error: Algoritmo '{algoritmo}' no reconocido en JSON." + Fore.RESET)
                sys.exit(1)
        else:
            print(f"No hay suficientes datos para analizar críticas {sentimiento}s.")
            
    print(Fore.YELLOW + "\n=== Proceso completado. ===" + Fore.RESET)