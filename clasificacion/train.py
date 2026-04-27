# -*- coding: utf-8 -*-  

# ======================= PLANTILLA - TRAIN ======================= 


"""
Autores: Daniel Talmaci & June Castro 
Script para la implementación de los siguientes algoritmos:
        1. kNN
        2. Decision Tree
        3. Random Forest
        4. Naïve Bayes
"""


# ======================= IMPORTS =======================  
import emoji 
import random  # Permite generar números aleatorios.
import sys  # Permite interactuar con el sistema, por ejemplo salir del programa.
import signal  # Permite capturar señales del sistema, como Ctrl+C.
import argparse  # Sirve para leer argumentos que se pasan por terminal.
import pandas as pd  # Librería para trabajar con tablas de datos (DataFrames).
import numpy as np  # Librería para operaciones numéricas y arrays.
import string  # Incluye constantes y utilidades para trabajar con texto.
import pickle  # Permite guardar y cargar objetos Python en archivos.
import time  # Permite medir tiempos y hacer pausas.
import json  # Permite leer y escribir archivos JSON.
import csv  # Permite escribir y leer archivos CSV.
import os  # Permite trabajar con carpetas, rutas y archivos del sistema.
from colorama import Fore  # Permite imprimir texto en colores por terminal.
from tqdm import tqdm  # Librería para mostrar barras de progreso en terminal.

# Sklearn 

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB # Importa el clasificador Naive Bayes. 
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV



from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
# Escaladores para normalizar o reescalar variables numéricas.

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# TfidfVectorizer convierte texto a números usando TF-IDF.
# CountVectorizer convierte texto a números contando palabras (BOW).

from sklearn.svm import SVC #Clasificador SVM 
from sklearn.linear_model import LogisticRegression #Clasificador Logistic Regression
from sklearn.neighbors import KNeighborsClassifier  # Clasificador kNN.
from sklearn.tree import DecisionTreeClassifier  # Clasificador árbol de decisión.
from sklearn.ensemble import RandomForestClassifier  # Clasificador Random Forest.
from sklearn.preprocessing import OneHotEncoder  # Codifica variables categóricas como variables numéricas binarias.

# Nltk 

import nltk  # Librería de procesamiento de lenguaje natural.
from nltk.corpus import stopwords  # Lista de palabras vacías, como "the", "is", etc.
from nltk.stem import PorterStemmer  # Herramienta para hacer stemming (lematizar).
from nltk.tokenize import word_tokenize  # Función para dividir texto en tokens/palabras.

# Imblearn  

from imblearn.under_sampling import RandomUnderSampler  # Para hacer undersampling.
from imblearn.over_sampling import RandomOverSampler  # Para hacer oversampling.
from imblearn.over_sampling import SMOTE, ADASYN  # Técnicas avanzadas de oversampling.


# ======================= PROGRAMA =======================  # Separador decorativo principal.

package = {}

# ----------------- Funciones auxiliares -----------------  # Separador de funciones auxiliares.

def signal_handler(sig, frame): 
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """

    print("\nSaliendo del programa...") 
    sys.exit(0)  

def parse_args(): 
    """
    Función para parsear los argumentos de entrada
    """

    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")

    # Parametros necesarios
    parse.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)

    # Parametros opcionales
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
  
    args = parse.parse_args()
    # Lee los argumentos realmente escritos en la terminal y los guarda en args.

    try:  
        with open(args.json, 'r') as json_file:  # Abre el archivo JSON indicado en modo lectura.
            config = json.load(json_file)  # Carga el contenido del JSON como diccionario Python.

        for key, value in config.items():  # Recorre todas las claves y valores del JSON.
            setattr(args, key, value)  # Añade cada clave del JSON como atributo dentro de args.

        if hasattr(args, "data_file"):
            # Comprueba si args tiene data_file.
            args.file = args.data_file

    except FileNotFoundError: 
        print(f"Error: No se encontró el archivo {args.json}")  
        sys.exit(1)  

    return args 

def load_data(file):  
    """
    Función para cargar los datos de un fichero csv
    Param
        file: Fichero csv
    Returns:
        Datos del fichero
    """

    try: 
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)

        return data


    except Exception as e:  
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)

        print(e) 
        sys.exit(1) 

def convert_score_to_sentiment(df, target_column):
    """
    Convierte score (1-5) a:
    NEGATIVO / NEUTRO / POSITIVO
    """

    def map_score(value):
        if pd.isna(value):
            return value

        try:
            value = int(value)
        except:
            return value

        if value in [1, 2]:
            return "NEGATIVO"
        elif value == 3:
            return "NEUTRO"
        elif value in [4, 5]:
            return "POSITIVO"
        else:
            return value

    df[target_column] = df[target_column].apply(map_score)

    return df

# ---------- Funciones para cálculo de Métricas ----------  

def calculate_fscore(y_true, y_pred):  # Función para calcular el F1-score.
    """
    Función para calcular el F-Score del modelo según la configuración indicada.
    """
    average_type = args.metrics.get("fscore_average", "none")
    # Lee del JSON el tipo de media que se quiere usar: micro, macro o none.

    if average_type == "micro": 
        return f1_score(y_true, y_pred, average='micro')
        # Devuelve el F1 micro.

    elif average_type == "macro":  
        return f1_score(y_true, y_pred, average='macro')
        # Devuelve el F1 macro.

    elif average_type == "none":  
        return f1_score(y_true, y_pred)
        # Devuelve el F1 por defecto de sklearn.

    else: 
        raise ValueError(f"Valor no válido para fscore_average: {average_type}")
    
def calculate_precision(y_true, y_pred):  
    """
    Función para calcular la precisión del modelo según la configuración indicada.
    """
    average_type = args.metrics.get("fscore_average", "none")
    # Lee el tipo de media configurado.

    if average_type == "micro":  
        return precision_score(y_true, y_pred, average='micro', zero_division=0)
        # Devuelve precision micro. Si hay división entre 0, devuelve 0.

    elif average_type == "macro":  
        return precision_score(y_true, y_pred, average='macro', zero_division=0)
        # Devuelve precision macro.

    elif average_type == "none":  
        return precision_score(y_true, y_pred, zero_division=0)
        # Devuelve precision normal.

    else:  
        raise ValueError(f"Valor no válido para fscore_average: {average_type}")


def calculate_recall(y_true, y_pred):  
    """
    Función para calcular el recall del modelo según la configuración indicada.
    """
    average_type = args.metrics.get("fscore_average", "none")
    # Lee el tipo de media configurado.

    if average_type == "micro":  
        return recall_score(y_true, y_pred, average='micro', zero_division=0)
        # Devuelve recall micro.

    elif average_type == "macro":
        return recall_score(y_true, y_pred, average='macro', zero_division=0)
        # Devuelve recall macro.

    elif average_type == "none": 
        return recall_score(y_true, y_pred, zero_division=0)
        # Devuelve recall normal.

    else:  
        raise ValueError(f"Valor no válido para fscore_average: {average_type}")
        # Lanza error.

def calculate_confusion_matrix(y_true, y_pred): 
    """
    Función para calcular la matriz de confusión

    Returns:
        Matriz de confusión
    """
    return confusion_matrix(y_true, y_pred)

def calculate_classification_report(y_true, y_pred):  
    """
    Función para calcular el informe de clasificación

    Returns:
        Informe de clasificación
    """
    return classification_report(y_true, y_pred)
    
def get_scoring_metrics():  
    """
    Devuelve las métricas de scoring para GridSearchCV según la configuración del JSON.
    """
    average_type = args.metrics.get("fscore_average", "none")


    if average_type == "micro":  
        return {
            "score": args.estimator, 
            "precision": "precision_micro",  
            "recall": "recall_micro" 
        }

    elif average_type == "macro":  
        return {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        }

    elif average_type == "none": 
        return {
            "score": args.estimator, 
            "precision": "precision",  
            "recall": "recall" 
        }

    else:  
        raise ValueError(f"Valor no válido para fscore_average: {average_type}")

# -------------- Funciones de Preprocesado -------------- 

def select_features(df):  
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.
    """
    global package
    try:  
        numerical_feature = df.select_dtypes(include=['int64', 'float64']).copy()
        # Selecciona columnas numéricas enteras o decimales y hace una copia.

        categorical_feature = df.select_dtypes(include='object').copy()
        # Selecciona todas las columnas de tipo texto/object y hace una copia.

        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
        # De esas columnas de texto, considera categóricas solo las que tengan pocos valores únicos.

        text_feature = df.select_dtypes(include='object').drop(columns=categorical_feature.columns, errors='ignore').copy()
        # Las columnas object que no sean categóricas se consideran de texto libre.

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
      

        if args.debug:  
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            # Imprime las columnas numéricas.

            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            # Imprime las columnas de texto.

            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
            # Imprime las columnas categóricas.

        package["numerical_columns"] = list(numerical_feature.columns)
        package["categorical_columns"] = list(categorical_feature.columns)
        package["text_columns"] = list(text_feature.columns)
        package["unique_category_threshold"] = args.preprocessing["unique_category_threshold"]

        return numerical_feature, text_feature, categorical_feature


    except Exception as e:  
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)

        print(e)  
        sys.exit(1) 

def build_scaler(method):  
    """
    Función que crea el scaler correspondiente al método indicado.
    """
    if method == "standard": 
        return StandardScaler()
        # Devuelve un StandardScaler.

    if method == "minmax":  
        return MinMaxScaler()
        # Devuelve un MinMaxScaler.

    if method == "maxabs":  
        return MaxAbsScaler()
        # Devuelve un MaxAbsScaler.

    if method in ["none", None]: 
        return None
        # Devuelve None.

    raise ValueError(f"Método de escalado no soportado: {method}")


def check_imbalance():  # Función para comprobar si el dataset está desbalanceado.
    """
    Comprueba si el dataset completo está desbalanceado según el umbral del JSON.
    """
    global data  

    balancing_cfg = get_balancing_config()

    method = balancing_cfg.get("method", "none")

    if method == "none":  
        print("No se comprobará balanceo porque el método configurado es 'none'")
        # Informa de que no se va a balancear.

        return False
        # Devuelve False, es decir, no se considera necesario balancear.

    threshold = balancing_cfg.get("imbalance_threshold", 0.8)
    # Obtiene el umbral de desbalanceo. Por defecto 0.8.

    class_counts = data[args.prediction].value_counts()
    # Cuenta cuántas muestras hay de cada clase en la variable objetivo.

    if len(class_counts) < 2:  # Si solo hay una clase...
        print("No se aplica balanceo porque solo hay una clase")
        return False


    majority_count = class_counts.max()
    # Obtiene el número de muestras de la clase mayoritaria.

    minority_count = class_counts.min()
    # Obtiene el número de muestras de la clase minoritaria.

    ratio = minority_count / majority_count
    # Calcula la relación entre clase minoritaria y mayoritaria.

    print("Distribución de clases antes del split:")
    print(class_counts)
    # Imprime el conteo por clase.

    print(f"Ratio minoritaria/mayoritaria: {ratio:.4f}")
    # Imprime el ratio con 4 decimales.

    if ratio < threshold:  # Si la relación es menor que el umbral...
        print("El dataset se considera desbalanceado")
        return True
        # Devuelve True.

    print("El dataset no se considera desbalanceado")
    # Si no, informa que no está desbalanceado.
    return False
    # Devuelve False.

def over_under_sampling(x_train, y_train):  # Función para balancear solo train.
    """
    Realiza oversampling o undersampling SOLO sobre train.
    """
    balancing_cfg = get_balancing_config()
    # Lee la configuración de balanceo.

    method = balancing_cfg.get("method", "none")
    # Obtiene el método.

    random_state = balancing_cfg.get("random_state", 42)
    # Obtiene semilla aleatoria para reproducibilidad.

    if method == "none":
        print("No se aplica balanceo en train porque el método configurado es 'none'")
        # Informa de ello.
        return x_train, y_train
        # Devuelve train sin cambios.

    if method == "oversampling":  
        sampler = SMOTE(random_state=random_state)
        print("Aplicando Oversampling tipo SMOTE SOLO en train...")


    elif method == "undersampling": 
        sampler = RandomUnderSampler(random_state=random_state)
        # Crea el objeto para reducir ejemplos de la clase mayoritaria.
        print("Aplicando Undersampling SOLO en train...")
    

    else:  
        raise ValueError(f"Método de balanceo no soportado: {method}")


    x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train)
    # Aplica el balanceo y devuelve nuevos datos de train balanceados.

    print("Balanceo aplicado correctamente sobre train")
    # Informa del éxito.
    print("Distribución de clases en y_train después del balanceo:")
    # Título informativo.
    print(pd.Series(y_train_resampled).value_counts())
    # Muestra el nuevo reparto de clases tras balancear.

    return x_train_resampled, y_train_resampled
    # Devuelve los nuevos datos de train.

def process_missing_values(x_train, x_dev, y_train, y_dev, numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en train y dev usando estadísticas calculadas SOLO con train.
    """
    global package
    
    missing_cfg = get_missing_config()
    # Lee la configuración de missing values.
    
    default_strategy = missing_cfg.get("default", {}).get("strategy", "none")

    per_column = missing_cfg.get("per_column", {})
    # Obtiene la configuración específica por columna.

    all_columns = list(numerical_feature.columns) + list(categorical_feature.columns)
    # Junta en una sola lista las columnas numéricas y categóricas.

    package["missing_values_info"] = {}
    package["missing_default_strategy"] = default_strategy

    for col in all_columns:
        if col in per_column:
            strategy_cfg = per_column[col]
        else:
            strategy_cfg = {"strategy": default_strategy}

        strategy = strategy_cfg.get("strategy", "none")
        is_numeric = col in numerical_feature.columns

        package["missing_values_info"][col] = {
            "strategy": strategy
        }


        if strategy == "drop_rows":  
            train_mask = x_train[col].notna()
            # Crea una máscara booleana con las filas de train que no son NaN.
            dev_mask = x_dev[col].notna()
            # Crea una máscara igual para dev.
            x_train = x_train.loc[train_mask].copy()
            # Filtra x_train dejando solo filas sin missing en esa columna.
            y_train = y_train.loc[train_mask].copy()
            # Filtra y_train de forma consistente.
            x_dev = x_dev.loc[dev_mask].copy()
            # Filtra x_dev.
            y_dev = y_dev.loc[dev_mask].copy()
            # Filtra y_dev.
            print(f"Se eliminan filas con missing en '{col}'")

        elif strategy == "mean" and is_numeric: 
            fill_value = x_train[col].mean()
            # Calcula la media SOLO con train.
            x_train[col] = x_train[col].fillna(fill_value)
            # Rellena los NaN de train con la media.
            x_dev[col] = x_dev[col].fillna(fill_value)
            # Rellena los NaN de dev con la media calculada en train.
            package["missing_values_info"][col]["fill_value"] = fill_value
            print(f"Se imputa la media en '{col}'")

        elif strategy == "median" and is_numeric:  
            fill_value = x_train[col].median()
            # Calcula la mediana en train.
            x_train[col] = x_train[col].fillna(fill_value)
            # Rellena train.
            x_dev[col] = x_dev[col].fillna(fill_value)
            # Rellena dev con el mismo valor.
            package["missing_values_info"][col]["fill_value"] = fill_value
            print(f"Se imputa la mediana en '{col}'")

        elif strategy == "mode":  
            if not x_train[col].mode().empty:
                # Comprueba que la moda exista.
                fill_value = x_train[col].mode()[0]
                # Toma el valor más frecuente.
                x_train[col] = x_train[col].fillna(fill_value)
                # Rellena train.
                x_dev[col] = x_dev[col].fillna(fill_value)
                # Rellena dev con el mismo valor.
                package["missing_values_info"][col]["fill_value"] = fill_value
                print(f"Se imputa la moda en '{col}'")

        elif strategy == "constant":  
            fill_value = strategy_cfg.get("value", 0 if is_numeric else "desconocido")
            # Toma el valor definido en JSON.
            # Si no hay valor, usa 0 si es numérica o "desconocido" si no lo es.
            x_train[col] = x_train[col].fillna(fill_value)
            # Rellena train.
            x_dev[col] = x_dev[col].fillna(fill_value)
            # Rellena dev.
            package["missing_values_info"][col]["fill_value"] = fill_value
            print(f"Se imputa un valor constante en '{col}': {fill_value}")

        elif strategy == "none": 
            print(f"No se aplica imputación en '{col}'")

        else: 
            raise ValueError(f"Estrategia de missing no válida para '{col}': {strategy}")

    return x_train, x_dev, y_train, y_dev
    # Devuelve train y dev ya tratados.

def expand_contractions(text):
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "can't": "can not",
        "couldn't": "could not",
        "won't": "will not",
        "wouldn't": "would not",
        "shouldn't": "should not"
    }

    for c, full in contractions.items():
        text = text.replace(c, full)

    return text


def simplify_text(x_train, x_dev, text_feature):
    """
    Simplifica el texto en train y dev.
    """
    global package
    print("Simplificando texto...")
  
    language = args.preprocessing.get("language", "english")
    stop_words = set(stopwords.words(language))
    palabras_protegidas = {
        "no", "not", "nor", "but", "against", "very"
    }

    stop_words = stop_words - palabras_protegidas

    stemmer = PorterStemmer()
    # Crea el objeto para hacer stemming.

    package["text_simplification"] = {
        "language": language,
        "lowercase": True,
        "remove_punctuation": True,
        "remove_stopwords": True,
        "stemming": True
    }

    def procesar_texto(texto):  # Función interna para procesar una cadena de texto.
        texto = emoji.demojize(texto)
        tokens = word_tokenize(texto)
        # Divide el texto en palabras/tokens.
        tokens = [t for t in tokens if t not in stop_words]
        # Elimina las stopwords.
        tokens = [stemmer.stem(t) for t in tokens]
        # Aplica stemming a cada token.
        return " ".join(tokens)
        # Vuelve a unir los tokens en una cadena.

    for col in text_feature.columns:  # Recorre cada columna de texto.
        print(f"Procesando columna {col}...")
        # Informa de qué columna está tratando.
        x_train[col] = x_train[col].fillna("")
        # Sustituye NaN por cadena vacía en train.
        x_dev[col] = x_dev[col].fillna("")
        # Sustituye NaN por cadena vacía en dev.
        x_train[col] = x_train[col].str.lower()
        x_train[col] = x_train[col].apply(expand_contractions)  
        # Convierte a minúsculas el texto de train.
        x_dev[col] = x_dev[col].str.lower()
        x_dev[col] = x_dev[col].apply(expand_contractions)  # ← TAMBIÉN AQUÍ
        # Convierte a minúsculas el texto de dev.
        x_train[col] = x_train[col].str.translate(str.maketrans('', '', string.punctuation))
        # Elimina signos de puntuación en train.
        x_dev[col] = x_dev[col].str.translate(str.maketrans('', '', string.punctuation))
        # Elimina signos de puntuación en dev.
        x_train[col] = x_train[col].apply(procesar_texto)
        # Aplica la función de tokenizar, quitar stopwords y stemmizar a train.
        x_dev[col] = x_dev[col].apply(procesar_texto)
        # Hace lo mismo en dev.

    return x_train, x_dev
    # Devuelve train y dev con texto simplificado.

def cat2num(x_train, x_dev, categorical_feature):  
    """
    Convierte las características categóricas en características numéricas utilizando One-Hot Encoding.
    """
    global package
    if categorical_feature.columns.size == 0:  # Si no hay columnas categóricas...
        package["categorical_encoder"] = None
        package["encoded_categorical_columns"] = []
        return x_train, x_dev
        # Devuelve los datos tal cual.

    print("Conversión de variables categóricas a numéricas (One-Hot Encoding)")
    
    encoder = OneHotEncoder(sparse_Modelos=False, handle_unknown="ignore")
    # Crea el encoder.
    # sparse=False hace que devuelva array normal en vez de matriz dispersa.
    # handle_unknown="ignore" ignora categorías nuevas en dev.

    encoded_train = encoder.fit_transform(x_train[categorical_feature.columns])
    # Ajusta el encoder con train y transforma esas columnas en variables dummy.
    encoded_dev = encoder.transform(x_dev[categorical_feature.columns])
    # Transforma dev usando el encoder ya ajustado con train.
    encoded_columns = encoder.get_feature_names_out(categorical_feature.columns)
    # Obtiene los nombres de las nuevas columnas generadas.
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_columns, index=x_train.index)
    # Convierte el array codificado de train en DataFrame.
    encoded_dev_df = pd.DataFrame(encoded_dev, columns=encoded_columns, index=x_dev.index)
    # Convierte el array codificado de dev en DataFrame.
    x_train = x_train.drop(columns=categorical_feature.columns)
    # Elimina de x_train las columnas categóricas originales.
    x_dev = x_dev.drop(columns=categorical_feature.columns)
    # Elimina de x_dev las columnas categóricas originales.
    x_train = pd.concat([x_train, encoded_train_df], axis=1)
    # Une las nuevas columnas binarias al x_train original.
    x_dev = pd.concat([x_dev, encoded_dev_df], axis=1)
    # Une las nuevas columnas binarias al x_dev original.

    print("Nuevas columnas creadas:")

    for col in encoded_columns:  # Recorre cada nueva columna creada.
        print(col)  # La imprime.

    package["categorical_encoder"] = encoder
    package["encoded_categorical_columns"] = list(encoded_columns)

    return x_train, x_dev
    # Devuelve train y dev ya transformados.

def reescaler(x_train, x_dev, numerical_feature):  
    """
    Reescala las características numéricas usando la configuración del JSON.
    El scaler se ajusta con train y se aplica a train y dev.
    """
    global package
    scaling_cfg = get_scaling_config()
    # Obtiene configuración de escalado.

    default_method = scaling_cfg.get("default", "none")
    # Método por defecto para todas las columnas.

    per_column = scaling_cfg.get("per_column", {})
    # Métodos específicos para columnas concretas.

    package["scalers"] = {}
    package["scaling_config"] = {
        "default": default_method,
        "per_column": per_column
    }

    for col in numerical_feature.columns:  # Recorre todas las columnas numéricas.
        if col not in x_train.columns:  # Si esa columna ya no está...
            continue
            # La salta.

        method = per_column.get(col, default_method)
        # Toma el método específico de la columna o el por defecto.

        scaler = build_scaler(method)
        # Construye el escalador correspondiente.

        if scaler is None:
            print(f"No se escala la columna {col}")
            package["scalers"][col] = None

        else:  
            x_train[col] = scaler.fit_transform(x_train[[col]])
            # Ajusta el scaler con train y transforma la columna en train.

            x_dev[col] = scaler.transform(x_dev[[col]])
            # Usa el mismo scaler para transformar dev.

            package["scalers"][col] = scaler

            print(f"Columna {col} escalada con {method}")

    return x_train, x_dev
    # Devuelve train y dev escalados.

def process_text(x_train, x_dev, text_feature):  # Función para vectorizar texto.
    """
    Procesa las características de texto utilizando TF-IDF o BOW.
    El vectorizador se ajusta con train y se aplica a train y dev.
    """
    global package
    try:  
        if text_feature.columns.size > 0:  
            text_train = x_train[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            # Une todas las columnas de texto de cada fila en un solo texto para train.

            text_dev = x_dev[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            # Hace lo mismo en dev.

            tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1,2),
                min_df=3,
                max_features=2000
            )
            
            tfidf_train = tfidf_vectorizer.fit_transform(text_train)
            tfidf_dev = tfidf_vectorizer.transform(text_dev)

            train_text_df = pd.DataFrame(
                tfidf_train.toarray(),
                columns=tfidf_vectorizer.get_feature_names_out(),
                index=x_train.index
            )

            dev_text_df = pd.DataFrame(
                tfidf_dev.toarray(),
                columns=tfidf_vectorizer.get_feature_names_out(),
                index=x_dev.index
            )

            x_train = pd.concat([x_train, train_text_df], axis=1)
            x_dev = pd.concat([x_dev, dev_text_df], axis=1)

            x_train.drop(text_feature.columns, axis=1, inplace=True)
            x_dev.drop(text_feature.columns, axis=1, inplace=True)

            package["text_vectorizer"] = tfidf_vectorizer
            package["text_vectorizer_type"] = "tf-idf"

            print(Fore.GREEN + "Texto tratado con éxito usando TF-IDF" + Fore.RESET)
        return x_train, x_dev

    except Exception as e: 
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)

        print(e)  # Imprime el error exacto.
        sys.exit(1)  # Cierra el programa.

def drop_features(x_train, x_dev):  
    """
    Elimina las columnas especificadas del conjunto de datos.
    """
    global package
    try:  
        features_to_drop = args.preprocessing.get("drop_features", [])
        # Lee del JSON qué columnas hay que eliminar.
        x_train = x_train.drop(columns=features_to_drop, errors="ignore")
        # Elimina esas columnas de train. Si alguna no existe, la ignora.
        x_dev = x_dev.drop(columns=features_to_drop, errors="ignore")
        # Elimina esas columnas de dev.

        package["drop_features"] = features_to_drop

        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)

        return x_train, x_dev
        # Devuelve train y dev.

    except Exception as e:  
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)

        print(e)  # Imprime el error concreto.
        sys.exit(1)  # Sale del programa.


def get_missing_config(): 
    """
    Devuelve la configuración de tratamiento de valores perdidos.
    """
    return args.preprocessing.get("missing_values", {})
    # Devuelve el bloque missing_values del JSON, o un diccionario vacío si no existe.

def get_scaling_config():  
    """
    Devuelve la configuración de escalado.
    """
    return args.preprocessing.get("scaling", {})
    # Devuelve el bloque scaling del JSON, o vacío si no existe.

def get_balancing_config():  
    """
    Devuelve la configuración de balanceo.
    """
    return args.preprocessing.get("balancing", {})

def preprocesar_datos(x_train, x_dev, y_train, y_dev, is_imbalanced):  
    global package
    
    x_train, x_dev = drop_features(x_train, x_dev)
    numerical_feature, text_feature, categorical_feature = select_features(x_train)

    x_train, x_dev, y_train, y_dev = process_missing_values(
        x_train, x_dev, y_train, y_dev, numerical_feature, categorical_feature
    )

    x_train, x_dev = simplify_text(x_train, x_dev, text_feature)
    x_train, x_dev = cat2num(x_train, x_dev, categorical_feature)
    x_train, x_dev = reescaler(x_train, x_dev, numerical_feature)
    x_train, x_dev = process_text(x_train, x_dev, text_feature)

    if is_imbalanced:  # Si hay desbalanceo
        x_train, y_train = over_under_sampling(x_train, y_train)
        # Aplica balanceo solo al train.

    package["final_feature_columns"] = list(x_train.columns)
    package["prediction_column"] = args.prediction
    package["algorithm"] = args.algorithm
    package["preprocessing_config"] = args.preprocessing

    return x_train, x_dev, y_train, y_dev

def divide_data():  # Función para dividir los datos.
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento. (a predecir)
    - y_dev: Serie con las etiquetas de desarrollo. (a predecir)
    """
    global data  

    try:  
        X = data.drop(columns=[args.prediction])
        # Crea X quitando la columna objetivo.
        y = data[args.prediction]
        # Crea y con la columna objetivo.

        x_train, x_dev, y_train, y_dev = train_test_split(
            X,  # Variables de entrada.
            y,  # Variable objetivo.
            test_size=args.split["test_size"],  # Proporción para dev/test.
            random_state=args.split["random_state"],  # Semilla para reproducibilidad.
            stratify=y  # Mantiene la proporción de clases en train y dev.
        )

        print(Fore.GREEN + "Datos divididos con éxito" + Fore.RESET)
        # Informa de que la división ha salido bien.

        if args.debug:  # Si debug está activado...
            print(Fore.MAGENTA + "> Tamaño x_train:" + Fore.RESET, x_train.shape)
            # Imprime tamaño de x_train.
            print(Fore.MAGENTA + "> Tamaño x_dev:" + Fore.RESET, x_dev.shape)
            # Imprime tamaño de x_dev.
            print(Fore.MAGENTA + "> Tamaño y_train:" + Fore.RESET, y_train.shape)
            # Imprime tamaño de y_train.
            print(Fore.MAGENTA + "> Tamaño y_dev:" + Fore.RESET, y_dev.shape)
            # Imprime tamaño de y_dev.

        return x_train, x_dev, y_train, y_dev
        # Devuelve los cuatro conjuntos.

    except Exception as e:  
        print(Fore.RED + "Error al dividir los datos" + Fore.RESET)
        print(e)  # Muestra el error concreto.
        sys.exit(1)  # Termina el programa.

def build_model_name(gs, algorithm_name):
    """
    Construye un nombre de fichero usando el mejor algoritmo y sus hiperparámetros.
    """
    best_params = gs.best_params_

    if algorithm_name == "kNN":
        k = best_params.get("n_neighbors", "x")
        weights = best_params.get("weights", "x")
        metric = best_params.get("metric", "x")


        weights_map = {
            "uniform": "uni",
            "distance": "dist"
        }

        metric_map = {
            "euclidean": "eucl",
            "manhattan": "manh"
        }

        weights_short = weights_map.get(weights, str(weights))
        metric_short = metric_map.get(metric, str(metric))

        return f"kNN_k{k}_{weights_short}_{metric_short}"

    elif algorithm_name == "decision_tree":
        criterion = best_params.get("criterion", "x")
        max_depth = best_params.get("max_depth", "none")
        min_split = best_params.get("min_samples_split", "x")
        min_leaf = best_params.get("min_samples_leaf", "x")

        depth_str = "none" if max_depth is None else str(max_depth)

        return f"decision_tree_{criterion}_d{depth_str}_split{min_split}_leaf{min_leaf}"

    elif algorithm_name == "random_forest":
        n_estimators = best_params.get("n_estimators", "x")
        criterion = best_params.get("criterion", "x")
        max_depth = best_params.get("max_depth", "none")
        min_split = best_params.get("min_samples_split", "x")
        min_leaf = best_params.get("min_samples_leaf", "x")

        depth_str = "none" if max_depth is None else str(max_depth)

        return f"random_forest_{n_estimators}trees_{criterion}_d{depth_str}_split{min_split}_leaf{min_leaf}"
        
    elif algorithm_name == "logistic_regression":
        C = best_params.get("C", "x")
        penalty = best_params.get("penalty", "x")
        solver = best_params.get("solver", "x")
        max_iter = best_params.get("max_iter", "x")

        return f"logistic_regression_C{C}_{penalty}_{solver}_iter{max_iter}"
        
    elif algorithm_name == "svm":
        C = best_params.get("C", "x")
        kernel = best_params.get("kernel", "x")
        gamma = best_params.get("gamma", "x")

        return f"svm_C{C}_{kernel}_gamma{gamma}"
        
    elif algorithm_name == "naive_bayes":
        if "var_smoothing" in best_params:
            vs = best_params["var_smoothing"]
            return f"naive_bayes_gaussian_vs{vs}"

        elif "binarize" in best_params:
            alpha = best_params.get("alpha", "x")
            fit_prior = best_params.get("fit_prior", "x")
            binarize = best_params.get("binarize", "x")
            return f"naive_bayes_bernoulli_a{alpha}_fp{fit_prior}_bin{binarize}"

        else:
            alpha = best_params.get("alpha", "x")
            fit_prior = best_params.get("fit_prior", "x")
            return f"naive_bayes_multinomial_a{alpha}_fp{fit_prior}"

    else:
        return algorithm_name
        
        
def guardar_historial_experimento(gs, algorithm_name):
    """
    Guarda TODOS los modelos probados (GridSearch) en el historial.
    No duplica y ordena por F1 macro.
    """

    historial_path = "Historial/historial_modelos.csv"

    vectorizer = package.get("text_vectorizer", None)

    if vectorizer is not None:
        max_features = vectorizer.max_features
        ngram_range = vectorizer.ngram_range

        if ngram_range == (1, 1):
            tipo_ngramas = "unigramas"
        elif ngram_range == (1, 2):
            tipo_ngramas = "unigramas+bigramas"
        else:
            tipo_ngramas = str(ngram_range)

        usa_bigramas = ngram_range[1] >= 2
    else:
        max_features = None
        tipo_ngramas = "-"
        usa_bigramas = False

    balancing_cfg = args.preprocessing.get("balancing", {})
    balancing_method = balancing_cfg.get("method", "none")

    if balancing_method == "oversampling":
        tipo_oversampling = "smote"
    else:
        tipo_oversampling = "-"

    results = pd.DataFrame(gs.cv_results_)

    nuevas_filas = []

    for _, row in results.iterrows():
        nueva_fila = {
            "algoritmo": algorithm_name,
            "hiperparametros": str(row["params"]),
            "f1_macro": row["mean_test_f1_macro"],
            "max_features": max_features,
            "tipo_ngramas": tipo_ngramas,
            "usa_bigramas": usa_bigramas,
            "balancing_method": balancing_method,
            "tipo_oversampling": tipo_oversampling
        }
        nuevas_filas.append(nueva_fila)

    nuevas_filas_df = pd.DataFrame(nuevas_filas)

    # Cargar historial
    if os.path.exists(historial_path):
        historial = pd.read_csv(historial_path)
    else:
        historial = pd.DataFrame()

    # Evitar duplicados
    if not historial.empty:
        historial = pd.concat([historial, nuevas_filas_df], ignore_index=True)
        historial = historial.drop_duplicates(
            subset=["algoritmo", "hiperparametros", "max_features", "tipo_ngramas", "balancing_method"],
            keep="first"
        )
    else:
        historial = nuevas_filas_df

    # Ordenar por F1
    historial = historial.sort_values(by="f1_macro", ascending=False)

    # Guardar
    os.makedirs("Historial", exist_ok=True)
    historial.to_csv(historial_path, index=False)

    print(Fore.GREEN + f"Historial actualizado con TODOS los modelos en {historial_path}" + Fore.RESET)
def save_model(gs, algorithm_name):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.
    """
    global package
    try:
        package["model"] = gs

        model_name = build_model_name(gs, algorithm_name)

        balancing_cfg = get_balancing_config()
        method = balancing_cfg.get("method", "none")

        if method == "none":
            balance_folder = "SinBalanceado"
        else:
            balance_folder = "Balanceado"

        Modelos_dir = f"Modelos/{balance_folder}/{algorithm_name}"
        os.makedirs(Modelos_dir, exist_ok=True)
        os.makedirs(Modelos_dir, exist_ok=True)

        pkl_filename = f"{Modelos_dir}/{model_name}.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(package, file)

        print(Fore.CYAN + f"Modelo guardado con éxito: {pkl_filename}" + Fore.RESET)
        guardar_historial_experimento(gs, algorithm_name)
        csv_filename = f"{Modelos_dir}/{model_name}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Algoritmo', 'Params', 'Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro'])

            results = pd.DataFrame(gs.cv_results_)
            results = results.sort_values(by="mean_test_f1_macro", ascending=False)
            
            for _, row in results.iterrows():
                writer.writerow([
                    algorithm_name,
                    row["params"],
                    row.get("mean_test_accuracy", ""),
                    row.get("mean_test_precision_macro", ""),
                    row.get("mean_test_recall_macro", ""),
                    row.get("mean_test_f1_macro", "")
                ])
            

    except Exception as e:
        print(Fore.RED + "Error al guardar el modelo" + Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev): 
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score del clasificador en el conjunto de desarrollo.
    - Precision del clasificador en el conjunto de desarrollo.
    - Recall del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:  # Solo muestra resultados detallados si verbose está activado.
        y_pred = gs.predict(x_dev)
        # Hace predicciones sobre el conjunto de desarrollo.

        average_type = args.metrics.get("fscore_average", "none")
        # Lee el tipo de media configurado.
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        # Imprime la mejor combinación de hiperparámetros.
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        # Imprime el mejor score obtenido en GridSearchCV.

        if average_type == "micro":  # Si se usa micro...
            print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, y_pred))
            # Imprime F1 micro.
            print(Fore.MAGENTA+"> Precision micro:\n"+Fore.RESET, calculate_precision(y_dev, y_pred))
            # Imprime precision micro.
            print(Fore.MAGENTA+"> Recall micro:\n"+Fore.RESET, calculate_recall(y_dev, y_pred))
            # Imprime recall micro.

        elif average_type == "macro":  # Si se usa macro...
            print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, y_pred))
            # Imprime F1 macro.
            print(Fore.MAGENTA+"> Precision macro:\n"+Fore.RESET, calculate_precision(y_dev, y_pred))
            # Imprime precision macro.
            print(Fore.MAGENTA+"> Recall macro:\n"+Fore.RESET, calculate_recall(y_dev, y_pred))
            # Imprime recall macro.

        elif average_type == "none":  # Si se usa configuración normal...
            print(Fore.MAGENTA+"> F1-score:\n"+Fore.RESET, calculate_fscore(y_dev, y_pred))
            # Imprime F1.
            print(Fore.MAGENTA+"> Precision:\n"+Fore.RESET, calculate_precision(y_dev, y_pred))
            # Imprime precision.
            print(Fore.MAGENTA+"> Recall:\n"+Fore.RESET, calculate_recall(y_dev, y_pred))
            # Imprime recall.

        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, y_pred))
        # Imprime el classification report completo.
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, y_pred))
        # Imprime la matriz de confusión.

def kNN():  
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos
    """
    is_imbalanced = check_imbalance()
    # Comprueba si el dataset está desbalanceado.
    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide los datos.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev, is_imbalanced)
    # Aplica todo el preprocesado.

    if args.debug:  # Si modo debug está activado
        try:
            train_debug = x_train.copy()
            # Hace copia de x_train.
            train_debug[args.prediction] = y_train.values
            # Añade la columna objetivo al train procesado.
            dev_debug = x_dev.copy()
            # Hace copia de x_dev.
            dev_debug[args.prediction] = y_dev.values
            # Añade la columna objetivo al dev procesado.
            train_debug.to_csv('Modelos/train-processed.csv', index=False)
            # Guarda el train procesado.
            dev_debug.to_csv('Modelos/dev-processed.csv', index=False)
            # Guarda el dev procesado.
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)

        except Exception as e:  # Si hay error...
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
            print(e) 

    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:

        gs = GridSearchCV(
            KNeighborsClassifier(),  # Modelo base kNN.
            args.kNN,  # Hiperparámetros a probar, leídos del JSON.
            cv=5,  # Validación cruzada con 5 particiones.
            n_jobs=args.cpu,  # Número de CPUs a usar.
            scoring=get_scoring_metrics(),  # Métricas de evaluación.
            refit="f1_macro"  # Reentrena usando como métrica principal "score".
        )

        start_time = time.time()
        # Guarda el instante inicial.
        gs.fit(x_train, y_train)
        # Entrena el GridSearchCV.
        end_time = time.time()
        # Guarda el instante final.

        for i in range(100): 
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)

        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    # Calcula el tiempo total de entrenamiento.

    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    mostrar_resultados(gs, x_dev, y_dev)
    save_model(gs, "kNN")
    # Guarda el modelo y los resultados.

def decision_tree(): 
    """
    Función para implementar el algoritmo de árbol de decisión.
    """
    is_imbalanced = check_imbalance()
    # Comprueba si el dataset está desbalanceado.
    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide los datos.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev, is_imbalanced)
    # Aplica todo el preprocesado.

    if args.debug:  # Si debug activo
        try:
            train_debug = x_train.copy()
            # Copia x_train.
            train_debug[args.prediction] = y_train.values
            # Añade etiquetas.
            dev_debug = x_dev.copy()
            # Copia x_dev.
            dev_debug[args.prediction] = y_dev.values
            # Añade etiquetas.
            train_debug.to_csv('Modelos/train-processed.csv', index=False)
            # Guarda train procesado.
            dev_debug.to_csv('Modelos/dev-processed.csv', index=False)
            # Guarda dev procesado.
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)

        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
            print(e)


    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:

        gs = GridSearchCV(
            DecisionTreeClassifier(),
            args.decision_tree,
            # Parámetros del árbol sacados del JSON.
            cv=5,
            # 5 folds.
            n_jobs=args.cpu,
            # CPUs.
            scoring=get_scoring_metrics(),
            # Métricas.
            refit="f1_macro"
            # Reentrena con la métrica principal.
        )

        start_time = time.time()
        # Tiempo inicial.
        gs.fit(x_train, y_train)
        # Entrena.
        end_time = time.time()
        # Tiempo final.

        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random()*2)
            # Actualiza barra.

        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    # Calcula tiempo total.

    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    # Lo muestra.
    mostrar_resultados(gs, x_dev, y_dev)
    # Muestra resultados.
    save_model(gs, "decision_tree")
    # Guarda el modelo.

def random_forest():  
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    """
    is_imbalanced = check_imbalance()
    # Comprueba si el dataset está desbalanceado.
    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide los datos.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev, is_imbalanced)
    # Aplica todo el preprocesado.

    if args.debug:
        try:
            train_debug = x_train.copy()
            # Copia x_train.
            train_debug[args.prediction] = y_train.values
            # Añade y_train.
            dev_debug = x_dev.copy()
            # Copia x_dev.
            dev_debug[args.prediction] = y_dev.values
            # Añade y_dev.
            train_debug.to_csv('Modelos/train-processed.csv', index=False)
            # Guarda train.
            dev_debug.to_csv('Modelos/dev-processed.csv', index=False)
            # Guarda dev.
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
            print(e)

    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:

        gs = GridSearchCV(
            RandomForestClassifier(),
            # Modelo Random Forest.
            args.random_forest,
            # Hiperparámetros del JSON.
            cv=5,
            # 5 folds.
            n_jobs=args.cpu,
            # CPUs.
            scoring=get_scoring_metrics(),
            # Métricas.
            refit="f1_macro"
            # Métrica principal para refit.
        )

        start_time = time.time()
        # Tiempo inicial.
        gs.fit(x_train, y_train)
        # Entrena.
        end_time = time.time()
        # Tiempo final.

        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random()*2)

        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    # Calcula tiempo.

    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    # Lo muestra.
    mostrar_resultados(gs, x_dev, y_dev)
    # Muestra métricas.
    save_model(gs, "random_forest")
    # Guarda modelo.

def naive_bayes():
    """
    Función para implementar el algoritmo Naive Bayes.
    Permite elegir entre Gaussian, Multinomial y Bernoulli según el JSON.
    """
    is_imbalanced = check_imbalance()
    # Comprueba si el dataset está desbalanceado.
    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide los datos.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev, is_imbalanced)
    # Aplica todo el preprocesado.

    if args.debug:
        try:
            train_debug = x_train.copy()
            train_debug[args.prediction] = y_train.values

            dev_debug = x_dev.copy()
            dev_debug[args.prediction] = y_dev.values

            train_debug.to_csv('Modelos/train-processed.csv', index=False)
            dev_debug.to_csv('Modelos/dev-processed.csv', index=False)

            print(Fore.GREEN + "Datos preprocesados guardados con éxito" + Fore.RESET)

        except Exception as e:
            print(Fore.RED + "Error al guardar los datos preprocesados" + Fore.RESET)
            print(e)

    nb_type = args.naive_bayes.get("selected_model", "gaussian").lower()

    if nb_type == "gaussian":
        model = GaussianNB()
        param_grid = args.naive_bayes.get("gaussian", {
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        })

    elif nb_type == "multinomial":
        model = MultinomialNB()
        param_grid = args.naive_bayes.get("multinomial", {
            "alpha": [0.1, 0.5, 1.0],
            "fit_prior": [True, False]
        })

    elif nb_type == "bernoulli":
        model = BernoulliNB()
        param_grid = args.naive_bayes.get("bernoulli", {
            "alpha": [0.1, 0.5, 1.0],
            "fit_prior": [True, False],
            "binarize": [0.0]
        })

    else:
        raise ValueError(f"Tipo de Naive Bayes no soportado: {nb_type}")

    with tqdm(total=100, desc='Procesando naive bayes', unit='iter', leave=True) as pbar:

        gs = GridSearchCV(
            model,
            param_grid,
            cv=5,
            n_jobs=args.cpu,
            scoring=get_scoring_metrics(),
            refit="f1_macro"
        )

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)

        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time

    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    mostrar_resultados(gs, x_dev, y_dev)
    save_model(gs, "naive_bayes")
    
def logistic_regression():
    """
    Función para implementar Logistic Regression.
    Aunque se llame regresión, aquí se usa para clasificación.
    """
    is_imbalanced = check_imbalance()
    # Comprueba si el dataset está desbalanceado.
    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide los datos.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev, is_imbalanced)
    # Aplica todo el preprocesado.

    if args.debug:
        try:
            train_debug = x_train.copy()
            train_debug[args.prediction] = y_train.values

            dev_debug = x_dev.copy()
            dev_debug[args.prediction] = y_dev.values

            train_debug.to_csv('Modelos/train-processed.csv', index=False)
            dev_debug.to_csv('Modelos/dev-processed.csv', index=False)

            print(Fore.GREEN + "Datos preprocesados guardados con éxito" + Fore.RESET)

        except Exception as e:
            print(Fore.RED + "Error al guardar los datos preprocesados" + Fore.RESET)
            print(e)

    with tqdm(total=100, desc='Procesando logistic regression', unit='iter', leave=True) as pbar:

        gs = GridSearchCV(
            LogisticRegression(),
            args.logistic_regression,
            cv=5,
            n_jobs=args.cpu,
            scoring=get_scoring_metrics(),
            refit="f1_macro"
        )

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)

        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time

    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    mostrar_resultados(gs, x_dev, y_dev)
    save_model(gs, "logistic_regression")
    
def svm():
    """
    Función para implementar SVM (Support Vector Machine)
    """
    is_imbalanced = check_imbalance()
    # Comprueba si el dataset está desbalanceado.
    x_train, x_dev, y_train, y_dev = divide_data()
    # Divide los datos.

    x_train, x_dev, y_train, y_dev = preprocesar_datos(x_train, x_dev, y_train, y_dev, is_imbalanced)
    # Aplica todo el preprocesado.

    if args.debug:
        try:
            train_debug = x_train.copy()
            train_debug[args.prediction] = y_train.values

            dev_debug = x_dev.copy()
            dev_debug[args.prediction] = y_dev.values

            train_debug.to_csv('Modelos/train-processed.csv', index=False)
            dev_debug.to_csv('Modelos/dev-processed.csv', index=False)

            print(Fore.GREEN + "Datos preprocesados guardados con éxito" + Fore.RESET)

        except Exception as e:
            print(Fore.RED + "Error al guardar los datos preprocesados" + Fore.RESET)
            print(e)

    with tqdm(total=100, desc='Procesando SVM', unit='iter', leave=True) as pbar:

        gs = GridSearchCV(
            SVC(),
            args.svm,
            cv=5,
            n_jobs=args.cpu,
            scoring=get_scoring_metrics(),
            refit="f1_macro"
        )

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)

        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time

    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    mostrar_resultados(gs, x_dev, y_dev)
    save_model(gs, "svm")
# ======================= PROGRAMA PRINCIPAL =======================  

if __name__ == "__main__":  

    np.random.seed(42)

    print("=== Clasificador ===")

    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    print("\n- Creando carpeta Modelos...")

    try:
        os.makedirs('Modelos')
        print(Fore.GREEN+"Carpeta Modelos creada con éxito"+Fore.RESET)

    except FileExistsError:

        print(Fore.GREEN+"La carpeta Modelos ya existe"+Fore.RESET)

    except Exception as e:

        print(Fore.RED+"Error al crear la carpeta Modelos"+Fore.RESET)

        print(e)

        sys.exit(1)

    print("\n- Cargando datos...")

    data = load_data(args.file)
    # Carga el CSV de datos y lo guarda en la variable global data.
    
    
    data = data.dropna(subset=[args.prediction]).copy()
    data = convert_score_to_sentiment(data, args.prediction)

    print(Fore.GREEN + "Columna objetivo convertida a sentimiento" + Fore.RESET)
    print(data[args.prediction].value_counts())

    print("\n- Descargando diccionarios...")

    nltk.download('stopwords')
    # Descarga el recurso stopwords.
    nltk.download('punkt')
    # Descarga el recurso necesario para tokenizar.
    nltk.download('punkt_tab')

    nltk.download('wordnet')
    # Descarga wordnet, aunque en este código realmente no se usa después.

    print("\n- Ejecutando algoritmo...")
    # Mensaje informativo.

    if args.algorithm == "kNN":

        try:
            kNN()
            print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
            sys.exit(0)

        except Exception as e:
            print(e)

    elif args.algorithm == "decision_tree":

        try:
            decision_tree()
            print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
            sys.exit(0)

        except Exception as e:
            print(e)

    elif args.algorithm == "random_forest":

        try:
            random_forest()
            print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
            sys.exit(0)

        except Exception as e:
            print(e)

    elif args.algorithm == "naive_bayes":

        try:
            naive_bayes()
            print(Fore.GREEN+"Algoritmo naive bayes ejecutado con éxito"+Fore.RESET)
            sys.exit(0)

        except Exception as e:
            print(e)
            
    elif args.algorithm == "logistic_regression":

        try:
            logistic_regression()
            print(Fore.GREEN+"Algoritmo logistic regression ejecutado con éxito"+Fore.RESET)
            sys.exit(0)

        except Exception as e:
            print(e)
            
    elif args.algorithm == "svm":

        try:
            svm()
            print(Fore.GREEN+"Algoritmo SVM ejecutado con éxito"+Fore.RESET)
            sys.exit(0)

        except Exception as e:
            print(e)

    else:
        print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
        sys.exit(1)
