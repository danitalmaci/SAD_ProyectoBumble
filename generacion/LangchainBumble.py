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
    # EXAMPLES
    # ------------------------
    examples_1 = """
    Example 1:
    Text: it's such a amazing app to find intresting people.
    Sentiment: Positive

    Example 2:
    Text: Horrible matching algorithm, becomes unfavorable after a while, nearly impossible to get matches as an average male..
    Sentiment: Negative

    Example 3:
    Text: too much interest on money more. it's fun here but it it's cut short with many demands.
    Sentiment: Neutral
    """

    examples_few = """
    Example 1:
    Text: I really enjoy this app. it's easy to use and the prompts add a personal touch. this isn't a sex app it's a chance to meet someone friends, and potentially more.
    Sentiment: Positive

    Example 2:
    Text: one best platform to understand the feelings of others and to share the intimacy
    Sentiment: Positive

    Example 3:
    Text: create room for video calls and audio calls, it will help alot
    Sentiment: Positive

    Example 4:
    Text: App is not working properly
    Sentiment: Negative

    Example 5:
    Text: I have been banned from the app for no reason? Can this issue pls be resolved. The customer support isn't working either.
    Sentiment: Negative

    Example 6:
    Text: When it's free you get all kinds of likes to make u pay once u pay there's no likes lol
    Sentiment: Negative

    Example 7:
    Text: Need to pay to get to talk to many women
    Sentiment: Neutral

    Example 8:
    Text: I didn't use that much after using I will share my opinion
    Sentiment: Neutral

    Example 9:
    Text: there has to be a free mode for a short time before asking to subscribe
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
            IMPORTANT: Respond with EXACTLY ONE WORD, no punctuation, no extra text. 
            Valid responses are: Positive Negative Neutral

            Text: {text}

            Your response (one word only):"""
        elif args.shot == "1":
            template = """You are a sentiment classifier. Classify the sentiment of the following text. 
            IMPORTANT: Respond with EXACTLY ONE WORD, no punctuation, no extra text. 
            Valid responses are: Positive Negative Neutral

            Examples: {examples_1}

            Text: {text}

            Your response (one word only):"""
        elif args.shot == "few":
            template = """You are a sentiment classifier. Classify the sentiment of the following text. 
            IMPORTANT: Respond with EXACTLY ONE WORD, no punctuation, no extra text. 
            Valid responses are: Positive Negative Neutral

            Examples: {examples_few}

            Text: {text}

            Your response (one word only):"""

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
        You are an experienced writer specialized in creating realistic user-generated content for online platforms. Write a short opinion comment (between 20–50 words) about a dating app similar to Tinder or Bumble. The tone of the comment must be {sentiment}. Keep the SAME sentiment.
        Requirements:
            - The text must read like a real user sharing their personal experience.
            - Use natural, conversational English.   
            - Include at least 2–3 specific aspects of the app.
            - Avoid generic statements; include concrete details.
            - Write in first person.
            - Do NOT mention any real app names.
            - Do NOT mention the type of app (dating app, social app, etc.).
            - Do NOT use category references like "dating apps" or similar.
            - Only refer to it as "the app" or "this app".
            - End naturally.

        New opinion comment:"""

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
