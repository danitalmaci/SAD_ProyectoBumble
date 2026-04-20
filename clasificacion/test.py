# -*- coding: utf-8 -*-  

# ======================= PLANTILLA - TEST ======================= 

"""
Autores: Daniel Talmaci & June Castro
Script para el test de modelos de clasificación.
"""  

import sys  # Permite usar funciones del sistema, por ejemplo salir del programa con sys.exit().
import json  # Permite leer y trabajar con archivos de configuración en formato JSON.
import pickle  # Permite cargar objetos serializados, en este caso el modelo guardado.
import string  # Proporciona utilidades relacionadas con cadenas, como la puntuación.
import argparse  # Sirve para leer argumentos pasados por terminal.
import signal  # Permite capturar señales del sistema, como Ctrl+C.
import os  # Permite trabajar con rutas, carpetas y archivos del sistema operativo.
import pandas as pd  # Librería para trabajar con datos tabulares, como CSVs y DataFrames.
from colorama import Fore  # Permite imprimir texto en colores por terminal.

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder  # Herramientas de escalado y codificación.
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Herramientas para convertir texto en números.
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score  # Métricas para evaluar el modelo.

import nltk  # Librería de procesamiento del lenguaje natural.
from nltk.corpus import stopwords  # Lista de palabras vacías, por ejemplo "the", "and", etc.
from nltk.stem import PorterStemmer  # Algoritmo para reducir palabras a su raíz.
from nltk.tokenize import word_tokenize  # Para dividir el texto en palabras/tokens.


# ======================= CARGA DE CONFIGURACION =======================   configuración.

def signal_handler(sig, frame):  
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """

    print("\nSaliendo del programa...")  
    sys.exit(0)  

def parse_args():  # Define una función para leer los argumentos y el JSON de configuración.
    """
    Función para parsear los argumentos de entrada
    """

    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.") 

    # Parametros necesarios
    parse.add_argument("-j", "--json", help="Archivo de configuración JSON", required=True)  
    
    # Parametros opcionales
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)  # Argumento opcional para indicar el estimador.
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)  # Argumento opcional para indicar cuántas CPUs usar.
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")  # Argumento opcional tipo bandera; si se pone, verbose será True.
    
    args = parse.parse_args()  # Procesa los argumentos escritos en la terminal y los guarda en args.

    try: 
        with open(args.json, 'r') as json_file: 
            config = json.load(json_file)  # Carga el contenido del JSON como diccionario Python.

        for key, value in config.items():  # Recorre todas las claves y valores del JSON.
            setattr(args, key, value)  # Añade cada clave del JSON como atributo dentro de args.

    except FileNotFoundError:  
        print(f"Error: No se encontró el archivo {args.json}") 
        sys.exit(1)  

    return args  
    
    
    
    

# ======================= MODELO =======================  

def load_model(model_path):  
    """
    Carga el modelo desde el archivo indicado y lo devuelve.
    """
    try:  
        with open(model_path, 'rb') as file:  # Abre el archivo del modelo en modo binario lectura.
            package = pickle.load(file)  # Deserializa el contenido del archivo y lo guarda en package.
            print(Fore.GREEN + f"Modelo cargado con éxito desde {model_path}" + Fore.RESET)  
            return package  # Devuelve el paquete cargado.
            
    except Exception as e:  
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET) 
        print(e)  
        sys.exit(1)  
        


# ======================= FUNCIONES AUXILIARES =======================   

def get_missing_config(config):  # Devuelve la parte de la configuración relacionada con valores faltantes.
    return config.get("preprocessing", {}).get("missing_values", {})  # Busca preprocessing -> missing_values; si no existe, devuelve {}.

def get_scaling_config(config):  # Devuelve la parte de la configuración relacionada con el escalado.
    return config.get("preprocessing", {}).get("scaling", {})  # Busca preprocessing -> scaling; si no existe, devuelve {}.

def get_metrics_average(config):  # Devuelve el tipo de promedio que se usará para las métricas.
    return config.get("metrics", {}).get("fscore_average", "macro")  # Busca metrics -> fscore_average; si no existe, usa "macro".

def build_scaler(method):  # Crea y devuelve el scaler adecuado según el nombre recibido.
    if method == "standard":  # Si el método es standard...
        return StandardScaler()  
    if method == "minmax":  # Si el método es minmax...
        return MinMaxScaler() 
    if method == "maxabs":  # Si el método es maxabs...
        return MaxAbsScaler()  
    if method in ["none", None]:  # Si no se quiere escalado...
        return None  
    raise ValueError(f"Método de escalado no soportado: {method}")  # Si el método no es válido, lanza un error.
    

def select_features(df, config, package):  
    """
    Separa las características del conjunto de datos en numéricas, de texto y categóricas.
    """
    try:  
        numerical_columns = package.get("numerical_columns", [])  # Recupera del package la lista de columnas numéricas aprendidas en train.
        categorical_columns = package.get("categorical_columns", [])  # Recupera la lista de columnas categóricas aprendidas en train.
        text_columns = package.get("text_columns", [])  # Recupera la lista de columnas de texto aprendidas en train.

        numerical_feature = df[[col for col in numerical_columns if col in df.columns]].copy()  # Selecciona del DataFrame solo las columnas numéricas que existan realmente.
        categorical_feature = df[[col for col in categorical_columns if col in df.columns]].copy()  # Selecciona las columnas categóricas que existan.
        text_feature = df[[col for col in text_columns if col in df.columns]].copy()  # Selecciona las columnas de texto que existan.

        print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET) 
        print("Columnas numéricas:", list(numerical_feature.columns)) 
        print("Columnas categóricas:", list(categorical_feature.columns))  
        print("Columnas de texto:", list(text_feature.columns))  
        return numerical_feature, text_feature, categorical_feature  

    except Exception as e:  # Si algo falla durante la separación...
        print(Fore.RED + "Error al separar los datos" + Fore.RESET) 
        print(e)  
        sys.exit(1)  
        
        


# ======================= PREPROCESADO =======================  

def process_missing_values(data, numerical_feature, categorical_feature, config, package):  
    """
    Procesa los valores faltantes usando los valores aprendidos en train.
    """
    missing_info = package.get("missing_values_info", {})  # Recupera del package la información de cómo se trataron los missing values en train.

    for col in numerical_feature.columns:  # Recorre cada columna numérica.
        if col in data.columns and data[col].isnull().sum() > 0:  # Comprueba que la columna exista y que tenga valores nulos.
            if col in missing_info:  # Comprueba si en train se guardó información para esa columna.
                strategy = missing_info[col].get("strategy", "none")  # Recupera la estrategia usada en train para esa columna.

                if strategy == "drop_rows":  # Si la estrategia era eliminar filas...
                    data = data.dropna(subset=[col])  # ...borra las filas donde esa columna tenga nulo.
                    print(f"Se eliminan filas con missing en '{col}'")  # Informa de ello por pantalla.
                    
                elif strategy in ["mean", "median", "mode", "constant"]:  # Si la estrategia era imputar con algún valor...
                    fill_value = missing_info[col].get("fill_value", None)  # ...recupera el valor aprendido en train.
                    data[col] = data[col].fillna(fill_value)  # Sustituye los nulos por ese valor.
                    print(f"Se imputa en '{col}' el valor aprendido en train: {fill_value}")  # Informa del valor usado.
                    
                elif strategy == "none":  # Si no se debía hacer nada...
                    print(f"No se aplica imputación en '{col}'")  # ...lo indica por pantalla.
                    
            else:  # Si esa columna no aparece en la información guardada...
                print(f"No se aplica imputación en '{col}'")  # ...también indica que no se imputará.

    for col in categorical_feature.columns:  # Recorre cada columna categórica.
        if col in data.columns and data[col].isnull().sum() > 0:  # Comprueba que exista y tenga nulos.
            if col in missing_info:  # Si hay información guardada para esa columna...
                strategy = missing_info[col].get("strategy", "none")  # Recupera la estrategia usada en train.

                if strategy == "drop_rows":  # Si había que eliminar filas...
                    data = data.dropna(subset=[col])  # ...elimina filas con nulos en esa columna.
                    print(f"Se eliminan filas con missing en '{col}'")  # Informa de ello.
                elif strategy in ["mode", "constant", "mean", "median"]:  # Si había que imputar...
                    fill_value = missing_info[col].get("fill_value", None)  # Recupera el valor guardado en train.
                    data[col] = data[col].fillna(fill_value)  # Rellena los nulos con ese valor.
                    print(f"Se imputa en '{col}' el valor aprendido en train: {fill_value}")  # Informa del valor usado.
                elif strategy == "none":  # Si no había que hacer nada...
                    print(f"No se aplica imputación en '{col}'")  # ...lo indica.
            else:  # Si no hay configuración para esa columna...
                print(f"No se aplica imputación en '{col}'")  # ...tampoco aplica nada.

    return data  # Devuelve el DataFrame tras tratar los valores faltantes.

def simplify_text(data, text_feature, config):
    """
    Simplifica el texto: minúsculas, quitar puntuación, tokenizar, eliminar stopwords y stemming.
    """

    language = config.get("preprocessing", {}).get("language", "english")
    stop_words = set(stopwords.words(language))
    stemmer = PorterStemmer()  # Crea el objeto que hará stemming.


    def procesar_texto(texto):  
        tokens = word_tokenize(texto)  # Divide el texto en palabras o tokens.
        tokens = [t for t in tokens if t not in stop_words]  # Elimina las palabras vacías.
        tokens = [stemmer.stem(t) for t in tokens]  # Reduce cada palabra a su raíz.
        return " ".join(tokens)  # Une de nuevo los tokens en una sola cadena de texto.

    for col in text_feature.columns:  # Recorre cada columna de texto.

        data[col] = data[col].fillna("")  # Sustituye posibles nulos por cadenas vacías.
        data[col] = data[col].str.lower()  # Convierte todo el texto a minúsculas.
        data[col] = data[col].str.translate(str.maketrans('', '', string.punctuation))  # Elimina signos de puntuación.
        data[col] = data[col].apply(procesar_texto)  # Aplica la función interna de tokenización, stopwords y stemming.

    return data  # Devuelve el DataFrame con el texto ya simplificado.
    

def cat2num(data, categorical_feature, package):  
    """
    Convierte las variables categóricas en numéricas con One-Hot Encoding.
    """
    if categorical_feature.columns.size == 0:  # Si no hay columnas categóricas...
        return data 

    encoder = package.get("categorical_encoder", None)  # Recupera del package el codificador entrenado en train.

    if encoder is None:  # Si no existe encoder guardado...
        return data  

    encoded = encoder.transform(data[categorical_feature.columns])  # Aplica el encoder a las columnas categóricas del test.

    encoded_columns = encoder.get_feature_names_out(categorical_feature.columns)  # Obtiene los nombres de las nuevas columnas generadas.
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=data.index)  # Crea un DataFrame con los datos codificados y el mismo índice.

    data = data.drop(columns=categorical_feature.columns)  # Elimina del DataFrame original las columnas categóricas antiguas.
    data = pd.concat([data, encoded_df], axis=1)  # Añade al DataFrame las nuevas columnas binarias generadas.
    return data  # Devuelve el DataFrame ya transformado.


def reescaler(data, numerical_feature, config, package): 
    """
    Reescala las características numéricas usando los scalers aprendidos en train.
    """
    scalers = package.get("scalers", {})  # Recupera del package el diccionario de scalers entrenados por columna.

    for col in numerical_feature.columns:  # Recorre cada columna numérica.
        if col not in data.columns:  # Si la columna ya no está en el DataFrame...
            continue  # ...salta a la siguiente.

        scaler = scalers.get(col, None)  # Recupera el scaler correspondiente a esa columna.

        if scaler is None:  # Si no hay scaler guardado para esa columna...
            print(f"No se escala la columna {col}")  # ...lo indica por pantalla.
        else:  # Si sí existe un scaler...
            data[col] = scaler.transform(data[[col]])  # ...transforma esa columna con el mismo scaler usado en train.
            print(f"Columna {col} escalada con el scaler aprendido en train")  # Informa de que se ha escalado.

    return data  # Devuelve el DataFrame con las columnas numéricas reescaladas.

def process_text(data, text_feature, config, package):  # Función para convertir el texto a variables numéricas.
    """
    Procesa las características de texto utilizando TF-IDF o BOW.
    """
    try:  # Intenta procesar el texto.
        if text_feature.columns.size > 0:  # Comprueba si realmente existen columnas de texto.
            text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)  # Une en una sola cadena el contenido de todas las columnas de texto de cada fila.

            vectorizer = package.get("text_vectorizer", None)  # Recupera el vectorizador entrenado en train.
            vectorizer_type = package.get("text_vectorizer_type", "none")  # Recupera el tipo de vectorizador usado: tf-idf, bow o none.

            if vectorizer_type == "tf-idf" and vectorizer is not None:  # Si el tipo es TF-IDF y existe el vectorizador...
                tfidf_matrix = vectorizer.transform(text_data)  # ...transforma el texto a su representación TF-IDF.

                text_features_df = pd.DataFrame(  # Crea un DataFrame con la matriz resultante.
                    tfidf_matrix.toarray(),  # Convierte la matriz dispersa a un array normal.
                    columns=vectorizer.get_feature_names_out(),  # Usa como nombres de columnas las palabras generadas por el vectorizador.
                    index=data.index  # Mantiene el mismo índice que el DataFrame original.
                )

                data = pd.concat([data, text_features_df], axis=1)  # Añade las columnas numéricas del TF-IDF al DataFrame original.
                data.drop(text_feature.columns, axis=1, inplace=True)  # Elimina las columnas de texto originales, ya transformadas.

                print(Fore.GREEN + "Texto tratado con éxito usando TF-IDF" + Fore.RESET)  # Mensaje de éxito en verde.

            elif vectorizer_type == "bow" and vectorizer is not None:  # Si el tipo es bolsa de palabras y existe el vectorizador...
                bow_matrix = vectorizer.transform(text_data)  # ...transforma el texto con BOW.

                text_features_df = pd.DataFrame(  # Crea un DataFrame con la matriz BOW.
                    bow_matrix.toarray(),  # Convierte la matriz dispersa a array.
                    columns=vectorizer.get_feature_names_out(),  # Usa los nombres de palabras como columnas.
                    index=data.index  # Mantiene los índices originales.
                )

                data = pd.concat([data, text_features_df], axis=1)  # Añade las nuevas columnas al DataFrame.
                data.drop(text_feature.columns, axis=1, inplace=True)  # Elimina las columnas de texto originales.

                print(Fore.GREEN + "Texto tratado con éxito usando BOW" + Fore.RESET)  # Mensaje de éxito en verde.
            else:  # Si no hay vectorizador o el tipo no es válido...
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)  # Muestra aviso en amarillo.
        else:  # Si no hay columnas de texto...
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)  # Informa de ello.

        return data  # Devuelve el DataFrame tras el tratamiento del texto.

    except Exception as e:  # Si ocurre algún error durante el procesado...
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)  # Muestra un mensaje de error.
        print(e)  # Muestra el detalle del error.
        sys.exit(1)  # Finaliza el programa.

def drop_features(data, config):  # Función para eliminar columnas que no se quieren usar.
    """
    Elimina las columnas especificadas del conjunto de datos.
    """
    try:  # Intenta eliminar las columnas indicadas.
        data = data.drop(columns=package.get("drop_features", []), errors="ignore")  # Elimina las columnas guardadas en package["drop_features"]; si alguna no existe, la ignora.
        print(Fore.GREEN + "Columnas eliminadas con éxito" + Fore.RESET)  # Mensaje de éxito.
        return data  # Devuelve el DataFrame sin esas columnas.
    except Exception as e:  # Si falla el borrado...
        print(Fore.RED + "Error al eliminar columnas" + Fore.RESET)  # Muestra error en rojo.
        print(e)  # Muestra el detalle del error.
        sys.exit(1)  # Termina el programa.

def align_features_to_model(X_test, package):  # Función para asegurar que las columnas de test coincidan exactamente con las del modelo.
    """
    Alinea las columnas de X_test con las columnas que espera el modelo.
    """
    expected_features = package.get("final_feature_columns", None)  # Recupera del package la lista exacta de columnas que espera el modelo.

    if expected_features is None:  # Si no existe esa lista...
        print(Fore.YELLOW + "No se han encontrado nombres de columnas esperadas en el modelo. No se alinean columnas." + Fore.RESET)  # Avisa de que no puede alinear.
        return X_test  # Devuelve X_test tal cual.

    for col in expected_features:  # Recorre cada columna que el modelo espera.
        if col not in X_test.columns:  # Si esa columna no existe en X_test...
            X_test[col] = 0  # ...la crea rellena de ceros.

    extra_cols = [col for col in X_test.columns if col not in expected_features]  # Calcula qué columnas sobran en X_test.
    if extra_cols:  # Si hay columnas extra...
        X_test = X_test.drop(columns=extra_cols)  # ...las elimina.

    X_test = X_test[expected_features]  # Reordena las columnas exactamente en el orden esperado por el modelo.

    print(Fore.GREEN + "Columnas alineadas con el modelo" + Fore.RESET)  # Muestra mensaje de éxito.
    return X_test  # Devuelve el conjunto de test alineado.

def preprocess_test_data(data, config, package, target_column=None):  # Función principal de preprocesado del conjunto de test.
    """
    Separa la columna objetivo, preprocesa el resto y devuelve X_test e y_real.
    """
    print("\n- Preprocesando datos de test...")  # Informa de que empieza el preprocesado.

    y_real = None  # Inicializa la variable donde se guardará la columna real objetivo, si existe.
    if target_column and target_column in data.columns:  # Comprueba si se ha indicado columna objetivo y si está en el DataFrame.
        y_real = data[target_column].copy()  # Guarda una copia de la columna objetivo real.
        data = data.drop(columns=[target_column])  # Elimina la columna objetivo de los datos de entrada para no usarla como característica.
        print(f"Columna objetivo '{target_column}' separada correctamente")  # Informa de que se ha separado bien.

    numerical_feature, text_feature, categorical_feature = select_features(data, config, package)  # Separa las columnas por tipos.

    data = process_missing_values(data, numerical_feature, categorical_feature, config, package)  # Trata los valores nulos siguiendo lo aprendido en train.
    data = simplify_text(data, text_feature, config)
    data = cat2num(data, categorical_feature, package)  # Convierte variables categóricas a numéricas.
    data = reescaler(data, numerical_feature, config, package)  # Aplica escalado a las columnas numéricas.
    data = process_text(data, text_feature, config, package)  # Convierte el texto a representación numérica.
    data = drop_features(data, config)  # Elimina columnas no deseadas.
    data = align_features_to_model(data, package)  # Alinea el DataFrame final con las columnas exactas del modelo.

    if y_real is not None:  # Si existe la columna real...
        y_real = y_real.loc[data.index]  # ...la reajusta al mismo índice que los datos finales por si se eliminaron filas.

    return data, y_real  # Devuelve X_test ya preparado y la columna real, si existe.

def calculate_metrics(y_real, predictions, config):  # Función para calcular métricas de evaluación.
    """
    Calcula las métricas usando la configuración del JSON.
    """
    average_type = get_metrics_average(config)  # Recupera de la configuración el tipo de promedio a usar.

    if average_type == "micro":  # Si se pide promedio micro...
        f1 = f1_score(y_real, predictions, average="micro")  # Calcula F1 micro.
        precision = precision_score(y_real, predictions, average="micro", zero_division=0)  # Calcula precisión micro.
        recall = recall_score(y_real, predictions, average="micro", zero_division=0)  # Calcula recall micro.
    elif average_type == "macro":  # Si se pide promedio macro...
        f1 = f1_score(y_real, predictions, average="macro")  # Calcula F1 macro.
        precision = precision_score(y_real, predictions, average="macro", zero_division=0)  # Calcula precisión macro.
        recall = recall_score(y_real, predictions, average="macro", zero_division=0)  # Calcula recall macro.
    else:  # En cualquier otro caso...
        f1 = f1_score(y_real, predictions, average="binary")  # Calcula F1 binario.
        precision = precision_score(y_real, predictions, average="binary", zero_division=0)  # Calcula precisión binaria.
        recall = recall_score(y_real, predictions, average="binary", zero_division=0)  # Calcula recall binario.

    return f1, precision, recall  # Devuelve las tres métricas calculadas.


# ======================= PROGRAMA PRINCIPAL =======================  

if __name__ == '__main__': 
    print("=== Clasificador === ")  

    signal.signal(signal.SIGINT, signal_handler)  
    args = parse_args()  
   
    config = vars(args)  

    input_cfg = config.get("input", {})  # Recupera la parte "input" del JSON; si no existe, usa diccionario vacío.
    output_cfg = config.get("output", {})  # Recupera la parte "output" del JSON; si no existe, usa diccionario vacío.

    file_path = input_cfg.get("file")  # Obtiene la ruta del CSV de entrada.
    model_path = input_cfg.get("model_path")  # Obtiene la ruta del archivo del modelo guardado.
    target_column = input_cfg.get("target", "")  # Obtiene el nombre de la columna objetivo, si se ha indicado.

    predictions_file = output_cfg.get("predictions_file", "output/predicciones.csv")  # Obtiene la ruta del archivo de salida para las predicciones.

    if not file_path:  # Comprueba que exista la ruta del archivo de entrada.
        print("Error: falta 'input.file' en el JSON")  # Muestra un error si no está definida.
        sys.exit(1)  # Sale del programa.

    if not model_path:  # Comprueba que exista la ruta al modelo.
        print("Error: falta 'input.model_path' en el JSON")  # Muestra un error si no está definida.
        sys.exit(1)  # Sale del programa.

    print("\n=== TEST === ")  # Muestra un encabezado para la fase de test.
    print("Fichero de entrada: ", file_path)  # Muestra el CSV que se va a usar.
    print("Modelo: ", model_path)  # Muestra la ruta del modelo cargado.
    print("Target: ", target_column if target_column else "(no especificado) ")  # Muestra la columna objetivo o indica que no se ha especificado.

    print("\n- Descargando diccionarios... ")  # Informa de que va a descargar recursos de NLTK.
    nltk.download('stopwords')  # Descarga el diccionario de stopwords.
    nltk.download('punkt')  # Descarga el tokenizador necesario para word_tokenize.
    nltk.download('wordnet')  # Descarga wordnet, aunque en este script concreto no se usa directamente.

    if not os.path.exists("output"):  # Comprueba si existe la carpeta de salida.
        os.makedirs("output")  # Si no existe, la crea.

    data_original = pd.read_csv(file_path)  # Lee el CSV de entrada y lo carga en un DataFrame.
    print("\nDatos cargados: ")  # Muestra un mensaje indicando que los datos ya están cargados.
    print(data_original.head())  # Enseña las primeras filas para comprobar que se ha leído bien.

    package = load_model(model_path)  # Carga el paquete del modelo guardado.
    model = package["model"]  # Extrae del package el modelo en sí.

    # 4. Pasar config (diccionario) a las funciones de preprocesado
    X_test, y_real = preprocess_test_data(data_original.copy(), config, package, target_column)  # Preprocesa los datos de test y separa la columna objetivo real.

    print("\n- Realizando predicciones... ")  # Informa de que empieza la predicción.
    predictions = model.predict(X_test)  # Usa el modelo para predecir las clases de X_test.

    results = data_original.copy()  # Crea una copia del DataFrame original para construir el archivo de resultados.

    if y_real is not None:  # Si existe columna objetivo real...
        results = results.loc[X_test.index].copy()  # ...ajusta results al mismo índice final que X_test.
        results[target_column + "_REAL"] = y_real.values  # Añade una columna con los valores reales.
    else:  # Si no hay columna objetivo real...
        results = results.loc[X_test.index].copy()  # ...solo ajusta el índice a las filas realmente usadas.

    results[target_column + "_PRED"] = predictions  # Añade una columna con las predicciones del modelo.

    print(Fore.GREEN + "Predicción realizada con éxito " + Fore.RESET)  # Informa de que la predicción ha terminado correctamente.

    if y_real is not None:  # Si hay valores reales, entonces se pueden calcular métricas.
        print("\n=== MÉTRICAS === ")  # Muestra un encabezado para las métricas.
        try:  # Intenta calcular y mostrar las métricas.
            f1, precision, recall = calculate_metrics(y_real, predictions, config)  # Calcula F1, precisión y recall.
            
            print("F1: ", f1)  # Muestra el valor de F1.
            print("Precision: ", precision)  # Muestra la precisión.
            print("Recall: ", recall)  # Muestra el recall.
            print("\nClassification report: ")  # Título para el informe completo de clasificación.
            print(classification_report(y_real, predictions))  # Muestra el classification report de sklearn.
            print("Matriz de confusión: ")  # Título para la matriz de confusión.
            print(confusion_matrix(y_real, predictions))  # Muestra la matriz de confusión.
        except Exception as e:  # Si falla el cálculo de métricas...
            print("No se han podido calcular las métricas: ", e)  # ...muestra el error, pero no detiene el programa.

    results.to_csv(predictions_file, index=False)  # Guarda el DataFrame de resultados en un CSV sin guardar el índice.
    print(Fore.GREEN + f"Predicciones guardadas en: {predictions_file} " + Fore.RESET)  # Informa de dónde se ha guardado el archivo final.
