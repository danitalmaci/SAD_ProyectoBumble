# -*- coding: utf-8 -*-  
import sys
import argparse
import pandas as pd
import string
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel  # ¡NUEVA IMPORTACIÓN!
from colorama import Fore
import emoji

# Sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Nltk 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# FUNCIONES AUXILIARES

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

# PREPROCESADO

def procesar_texto(texto, stop_words, stemmer):
    texto_con_emojis = emoji.demojize(str(texto), language='en')
    
    texto_con_emojis = texto_con_emojis.replace(":", " ").replace("_", " ")
    
    tokens = word_tokenize(texto_con_emojis.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    tokens = [stemmer.stem(t) for t in tokens if t.isalpha()]
    return " ".join(tokens)

# FUNCIONES CLUSTERING

def calcular_coherencia_lda(textos_tokenizados, corpus, id2word, sentimiento, args):
    """
    Entrena múltiples modelos LDA variando K y calcula su coherencia para encontrar el óptimo.
    """
    max_k = args.clustering.get("max_k", 10)
    print(Fore.CYAN + f"\n- [COHERENCIA LDA] Evaluando modelos desde K=2 hasta K={max_k} para críticas {sentimiento}..." + Fore.RESET)
    
    rango_k = range(3, max_k + 1)
    coherencias = []
    modelos = []

    for k in rango_k:
        modelo_lda = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=k,
            random_state=42,
            passes=args.clustering.get("lda_passes", 10),
            alpha=args.clustering.get("lda_alpha", 'symmetric')
        )
        modelos.append(modelo_lda)
        
        # Calcular coherencia c_v (se podría probar con otros en el JSON)
        coherencemodel = CoherenceModel(model=modelo_lda, texts=textos_tokenizados, dictionary=id2word, coherence='c_v')
        puntuacion = coherencemodel.get_coherence()
        coherencias.append(puntuacion)
        print(f"  * K={k} -> Coherencia: {puntuacion:.4f}")

    # Encontrar el índice con la coherencia más alta
    indice_optimo = coherencias.index(max(coherencias))
    k_optimo = rango_k[indice_optimo]
    mejor_modelo = modelos[indice_optimo]
    
    print(Fore.GREEN + f"  > ¡K óptimo encontrado de forma automática!: {k_optimo}" + Fore.RESET)

    # Crear gráfico de Coherencia
    plt.figure(figsize=(8, 5))
    plt.plot(rango_k, coherencias, marker='o', linestyle='-', color='g')
    plt.title(f'Optimización Coherencia LDA (c_v) - {sentimiento.capitalize()}')
    plt.xlabel('Número de Tópicos (K)')
    plt.ylabel('Score de Coherencia')
    plt.grid(True)
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_grafico = os.path.join(directorio_actual, 'output', f'coherencia_lda_{sentimiento}.png')
    plt.savefig(ruta_grafico)
    plt.close()

    return k_optimo, mejor_modelo

def ejecutar_lda_gensim(data, columna_texto_limpio, sentimiento, args):
    min_df = args.clustering.get("min_df", 3)
    max_df = args.clustering.get("max_df", 0.85)
    
    # Preparar el corpus
    textos_tokenizados = [str(texto).split() for texto in data[columna_texto_limpio]]
    id2word = corpora.Dictionary(textos_tokenizados)
    id2word.filter_extremes(no_below=min_df, no_above=max_df)
    corpus = [id2word.doc2bow(texto) for texto in textos_tokenizados]

    # Ejecutar la búsqueda automática del mejor modelo
    k_optimo, lda_model = calcular_coherencia_lda(textos_tokenizados, corpus, id2word, sentimiento, args)

    print(Fore.MAGENTA + f"\nTop Palabras por Tópico (Modelo Ganador K={k_optimo}):" + Fore.RESET)
    topicos_crudos = lda_model.show_topics(num_topics=k_optimo, num_words=args.clustering.get("top_words", 10), formatted=False)
    
    palabras_por_topico = {}
    for num_topico, palabras in topicos_crudos:
        lista_palabras = [palabra for palabra, peso in palabras]
        texto_topico = ", ".join(lista_palabras)
        palabras_por_topico[num_topico] = texto_topico
        print(f"  > Tópico {num_topico}: {texto_topico}")

    topicos_dominantes = []
    probabilidades = {i: [] for i in range(k_optimo)} # Diccionario para las nuevas columnas

    for bow in corpus:
        if len(bow) == 0:
            topicos_dominantes.append(-1)
            for i in range(k_optimo): probabilidades[i].append(0.0)
            continue
            
        # Obtenemos la distribución forzando a que devuelva probabilidad 0 si no pertenece
        distribucion = lda_model.get_document_topics(bow, minimum_probability=0.0)
        
        # 1. Sacamos el dominante
        topico_principal = sorted(distribucion, key=lambda x: x[1], reverse=True)[0][0]
        topicos_dominantes.append(topico_principal)
        
        # 2. Guardamos la probabilidad de CADA tópico para las columnas de Tableau
        # (Asegurándonos de que se guardan en el orden correcto)
        dict_dist = dict(distribucion)
        for i in range(k_optimo):
            probabilidades[i].append(round(dict_dist.get(i, 0.0), 4))

    data['Cluster_Dominante'] = topicos_dominantes
    # Añadimos las columnas al dataframe
    for i in range(k_optimo):
        data[f'Prob_Topico_{i}'] = probabilidades[i]

    data['Cluster_Dominante'] = topicos_dominantes
    data['Palabras_Clave'] = data['Cluster_Dominante'].map(palabras_por_topico)
    data.loc[data['Cluster_Dominante'] == -1, 'Palabras_Clave'] = "Sin datos"
    
    return data

def ejecutar_kmeans(data, columna_texto_limpio, sentimiento, args):
    diccionario_k = args.clustering.get("k_optimo_dict", {})
    k_optimo = diccionario_k.get(sentimiento, 4)
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
    return data

def guardar_resultados(data, nombre_archivo):
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(directorio_actual, 'output', nombre_archivo)
    data.to_csv(ruta_csv, index=False)
    print(Fore.GREEN + f"Datos guardados en {ruta_csv}" + Fore.RESET)

def metodo_del_codo(data, columna_texto_limpio, sentimiento, args):
    if not args.clustering.get("generar_codo", False):
        return

    print(Fore.CYAN + f"\n- [MÉTODO DEL CODO] Calculando inercias para {sentimiento}..." + Fore.RESET)
    
    vectorizer = TfidfVectorizer(max_df=args.clustering.get("max_df", 0.95), min_df=args.clustering.get("min_df", 2))
    X_tfidf = vectorizer.fit_transform(data[columna_texto_limpio])
    
    rango_k = range(2, args.clustering.get("max_k", 10) + 1)
    inercias = []

    for k in rango_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_tfidf)
        inercias.append(kmeans.inertia_)

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

# MAIN

if __name__ == "__main__":  
    print(Fore.YELLOW + "=== Pipeline de Topic Modeling & Clustering ===" + Fore.RESET)
    args = parse_args()

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    carpeta_output = os.path.join(directorio_actual, 'output')
    os.makedirs(carpeta_output, exist_ok=True)

    data = load_data(args.file)
    columna_texto = args.preprocessing["text_column"]
    columna_score = args.preprocessing.get("score_column", "score")
    
    print(Fore.CYAN + "- Clasificando sentimientos..." + Fore.RESET)
    data['sentimiento_analisis'] = data[columna_score].apply(clasificar_estrellas)
    
    print(Fore.CYAN + "- Aplicando NLTK (Stopwords y Stemming)..." + Fore.RESET)
    language = args.preprocessing.get("language", "english")
    
    try:
        stopwords.words(language)
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')

    stop_words = set(stopwords.words(language))
    stop_words.update(args.preprocessing.get("extra_stopwords", []))
    stemmer = PorterStemmer()

    # 1. Limpieza base
    data['texto_limpio'] = data[columna_texto].apply(lambda x: procesar_texto(x, stop_words, stemmer))

    print(Fore.CYAN + "- Generando bigramas..." + Fore.RESET)
    
    # Convertimos el texto limpio en una lista de listas de palabras
    textos_lista = [str(texto).split() for texto in data['texto_limpio']]
    
    min_count = args.preprocessing.get("bigram_min_count", 5) # La pareja debe aparecer al menos X veces en todo el dataset
    threshold = args.preprocessing.get("bigram_threshold", 10) # Nivel de "confianza" estadística para unirlas
    
    bigramas_detector = gensim.models.Phrases(textos_lista, min_count=min_count, threshold=threshold)
    bigramas_mod = gensim.models.phrases.Phraser(bigramas_detector)
    
    # Aplicamos los bigramas y volvemos a unir el texto en un string para el dataframe
    data['texto_limpio'] = [" ".join(bigramas_mod[doc]) for doc in textos_lista]

    # Leemos el algoritmo del JSON (por defecto 'hibrido' si no se especifica)
    algoritmo = args.clustering.get("algorithm", "hibrido").lower()
    
    data['num_palabras'] = data['texto_limpio'].apply(lambda x: len(str(x).split()))
    umbral = args.clustering.get("umbral_palabras_cortas", 5)

    resultados_totales = []

    for sentimiento in ['negativa', 'neutra', 'positiva']:
        print(Fore.YELLOW + f"\n=== ANALIZANDO SENTIMIENTO: {sentimiento.upper()} ===" + Fore.RESET)
        subset_data = data[data['sentimiento_analisis'] == sentimiento].copy()
        
        if len(subset_data) == 0:
            continue
        
        if algoritmo == "hibrido":
            textos_cortos = subset_data[subset_data['num_palabras'] <= umbral].copy()
            textos_largos = subset_data[subset_data['num_palabras'] > umbral].copy()

            print(Fore.CYAN + f"- Comentarios largos (> {umbral} palabras): {len(textos_largos)} -> LDA" + Fore.RESET)
            print(Fore.CYAN + f"- Comentarios cortos (<= {umbral} palabras): {len(textos_cortos)} -> K-MEANS" + Fore.RESET)

            if len(textos_largos) > 0:
                df_lda = ejecutar_lda_gensim(textos_largos, 'texto_limpio', f"{sentimiento}_largos", args)
                df_lda['Modelo'] = 'LDA'
                df_lda['Longitud'] = 'Largo'
                resultados_totales.append(df_lda)

            if len(textos_cortos) > 0:
                metodo_del_codo(textos_cortos, 'texto_limpio', f"{sentimiento}_cortos", args)
                df_kmeans = ejecutar_kmeans(textos_cortos, 'texto_limpio', f"{sentimiento}_cortos", args)
                df_kmeans['Modelo'] = 'K-Means'
                df_kmeans['Longitud'] = 'Corto'
                resultados_totales.append(df_kmeans)

        elif algoritmo == "lda":
            print(Fore.CYAN + f"- Todo el dataset ({len(subset_data)} reseñas) -> LDA" + Fore.RESET)
            if len(subset_data) > 0:
                df_lda = ejecutar_lda_gensim(subset_data, 'texto_limpio', f"{sentimiento}_todos", args)
                df_lda['Modelo'] = 'LDA'
                df_lda['Longitud'] = 'Todos'
                resultados_totales.append(df_lda)

        elif algoritmo == "kmeans":
            print(Fore.CYAN + f"- Todo el dataset ({len(subset_data)} reseñas) -> K-MEANS" + Fore.RESET)
            if len(subset_data) > 0:
                metodo_del_codo(subset_data, 'texto_limpio', f"{sentimiento}_todos", args)
                df_kmeans = ejecutar_kmeans(subset_data, 'texto_limpio', f"{sentimiento}_todos", args)
                df_kmeans['Modelo'] = 'K-Means'
                df_kmeans['Longitud'] = 'Todos'
                resultados_totales.append(df_kmeans)
                
        else:
            print(Fore.RED + f"Error: Algoritmo '{algoritmo}' no reconocido en el JSON. Usa 'lda', 'kmeans' o 'hibrido'." + Fore.RESET)
            sys.exit(1)

    if resultados_totales:
        df_final = pd.concat(resultados_totales, ignore_index=True)
        
        # Rellenar con ceros las columnas de probabilidad si existen (por K-Means)
        cols_probabilidades = [col for col in df_final.columns if 'Prob_Topico' in col]
        if cols_probabilidades:
            df_final[cols_probabilidades] = df_final[cols_probabilidades].fillna(0.0)
        
        # Guardar archivo con el nombre del algoritmo para no sobreescribir pruebas
        guardar_resultados(df_final, f'clustering_{algoritmo}_tableau.csv')       
    print(Fore.YELLOW + "\n=== Proceso completado. ===" + Fore.RESET)