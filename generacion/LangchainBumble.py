import argparse
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ------------------------
# MAIN
# ------------------------
def main():

    parser = argparse.ArgumentParser(description="Sentiment classification with Ollama")

    parser.add_argument('--mode', type=str, required=True, choices=['predict', 'oversample'])
    parser.add_argument('--model', type=str, default='llama3:8b-text-q2_K')
    parser.add_argument('--shot', type=str, default='0', choices=['0', '1', 'few'])

    # CSV
    parser.add_argument('--csv', type=str)
    parser.add_argument('--target', type=str, help='Columna de comentarios')
    parser.add_argument('--sentiment', type=str, default=None, help='Columna numérica 1-5')
    parser.add_argument('--score', type=str, default=None, help='Score de la opinión a generar en oversampling')
    parser.add_argument('--samples', type=int, default=10, help='Número de instancias a calsificar')

    args = parser.parse_args()

    model=load_model(args.mode, args.model)

    # ------------------------
    # FEW-SHOT EXAMPLES
    # ------------------------
    examples = """
    Example 1:
    Text: I love this app, it works perfectly.
    Sentiment: Positive

    Example 2:
    Text: This is the worst experience ever.
    Sentiment: Negative

    Example 3:
    Text: It is okay, nothing special.
    Sentiment: Neutral
    """

    # ------------------------
    # MODO 1: PREDICT
    # ------------------------
    if args.mode == "predict":

        # Validar argumentos obligatorios
        if args.csv is None:
            raise ValueError("You must provide --csv for predict mode")
        
        if args.sentiment is None:
            raise ValueError("You must provide --sentiment (ground truth column) for evaluation in predict mode")
        
        if args.target is None:
            raise ValueError("You must provide --target (text column) for predict mode")
        
        df = load_csv(args.csv, args.sentiment)

        if args.target not in df.columns:
            raise ValueError(f"Column '{args.target}' not found in CSV")

        if args.sentiment not in df.columns:
            raise ValueError(f"Column '{args.sentiment}' not found in CSV")
        
        df_sample = df.sample(n=min(args.samples, len(df)))

        if args.shot == "0":
            template = """You are a sentiment classifier. Classify the sentiment of the following text. 
            Respond with ONLY ONE word: 'Positive', 'Negative' or 'Neutral'. Text: "In my experience, a lot of matches never reply. 
            It takes away from the overall experience."

            Text: {text}

            Sentiment:"""
        elif args.shot == "1":
            template = """You are a sentiment classifier. Classify the sentiment of the following text. 
            Respond with ONLY ONE word: 'Positive', 'Negative' or 'Neutral'. Text: "In my experience, a lot of matches never reply. 
            It takes away from the overall experience."
            
            Examples: {examples}

            Text: {text}

            Sentiment:"""
        elif args.shot == "few":
            template = """You are a sentiment classifier. Classify the sentiment of the following text. 
            Respond with ONLY ONE word: 'Positive', 'Negative' or 'Neutral'. Text: "In my experience, a lot of matches never reply. 
            It takes away from the overall experience."
            
            Examples: {examples}

            Text: {text}

            Sentiment:"""

        prompt = PromptTemplate.from_template(template)
        chain = prompt | model

        predictions = []

        for text in df_sample[args.target]:
            try:
                ans = chain.invoke({'text': str(text)}).strip()
            except:
                ans = "ERROR"

            if ans not in ["Positive", "Negative", "Neutral"]:
                ans = "Neutral"

            predictions.append(ans)

        df_sample['prediction'] = predictions

        #------ Evaluación ------
        if args.sentiment is not None:

            y_true = df_sample['sentiment_mapped']
            y_pred = df_sample['prediction']

            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')

            print("\n--- MÉTRICAS ---")
            print(f"Accuracy: {round(acc,4)}")
            print(f"F1 Macro: {round(f1_macro,4)}\n")

            print("Classification Report:")
            print(classification_report(y_true, y_pred))

        output_file = "predictions.csv"
        df_sample.to_csv(output_file, index=False)

        print(f"Predictions saved in {output_file}")

    # ------------------------
    # MODO 2: OVERSAMPLING
    # ------------------------
    elif args.mode == "oversample":

        if args.score is None:
            raise ValueError("You must provide --score (1-5)")
        
        # Mapear el número a etiqueta
        sentiment_label = map_sentiment(args.score)

        template = """
        You are an expert data generator for Machine Learning training. 
        Generate a NEW, realistic user comment or review about the Instagram app that clearly expresses a {sentiment} sentiment.
        The comment must be natural, varied and strictly in English.

        Respond ONLY with the text of the generated comment."""

        prompt = PromptTemplate.from_template(template)
        chain = prompt | model

        new_rows = []

        for _ in range(args.samples):

            try:
                new_text = chain.invoke({'sentiment': sentiment_label}).strip()
            except:
                continue

            # limpieza básica
            if not new_text:
                continue

            new_rows.append({
                'content': new_text,
                'score': int(args.score)
            })

        df_new = pd.DataFrame(new_rows)

        output_file = "oversampled.csv"
        df_new.to_csv(output_file, index=False)

        print(f"Generated {len(new_rows)} new samples")
        print(f"New dataset saved to {output_file}")

    # ------------------------
    # EVALUACIÓN
    # ------------------------
    if args.mode == "predict":

        y_true = df_sample['sentiment_mapped']
        y_pred = df_sample['prediction']

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        print("\n--- MÉTRICAS ---")
        print(f"Accuracy: {round(acc,4)}")
        print(f"F1 Macro: {round(f1_macro,4)}\n")

        print("Classification Report:")
        print(classification_report(y_true, y_pred))

# ------------------------
# FUNCIÓN MAPEO (1–5 → sentimiento)
# ------------------------
def map_sentiment(x):
    try:
        x = int(x)
        if x <= 2:
            return "Negative"
        elif x == 3:
            return "Neutral"
        else:
            return "Positive"
    except:
        return "Neutral"
    
# ------------------------
# CARGAR CSV
# ------------------------
def load_csv(csv, sentiment):
    df = pd.read_csv(csv)

    # Si hay columna de rating → crear sentimiento
    if sentiment is not None:
        df['sentiment_mapped'] = df[sentiment].apply(map_sentiment)
    
    return df

# ------------------------
# MODELO
# ------------------------
def load_model(mode,name):
    if mode == "predict":
        model = OllamaLLM(
            model=name,           
            temperature=0,        # creatividad/aleatoriedad del modelo (1: creativo)
            num_predict=20,       # número máximo de tokens  
            repeat_penalty=1.1,   # penalización por repetir palabras o frases
            top_k=10,             # tamaño del vocabulario candidato
            top_p=0.5             # masa de probabilidad acumulada
        )
    else:
        model = OllamaLLM(
            model=name,  
            temperature=0.85,     # creatividad/aleatoriedad del modelo (1: creativo)
            num_predict=50,       # número máximo de tokens  
            repeat_penalty=1.15,  # penalización por repetir palabras o frases 
            top_k=40,             # tamaño del vocabulario candidato
            top_p=0.9             # masa de probabilidad acumulada
        )

    return model


# ------------------------
# EJECUCIÓN
# ------------------------
if __name__ == "__main__":
    main()
