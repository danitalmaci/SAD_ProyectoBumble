import argparse
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from sklearn.metrics import accuracy_score, classification_report, f1_score


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
# EVALUACIÓN
# ------------------------
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


# ------------------------
# MAIN
# ------------------------
def main():

    parser = argparse.ArgumentParser(description="Sentiment classification with Ollama")

    parser.add_argument('--mode', type=str, required=True, choices=['predict', 'oversample'])
    parser.add_argument('--model', type=str, default='llama3:8b-text-q2_K')
    parser.add_argument('--shot', type=str, default='0', choices=['0', '1', 'few'])

    # CSV
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--target', type=str, required=True, help='Columna de comentarios')
    parser.add_argument('--sentiment', type=str, default=None, help='Columna numérica 1-5')
    parser.add_argument('--samples', type=int, default=10)

    args = parser.parse_args()

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
    # MODELO
    # ------------------------
    model = OllamaLLM(
        model=args.model,
        temperature=0,
        num_predict=20,
        repeat_penalty=1.1,
        top_k=10,
        top_p=0.5
    )

    # ------------------------
    # CARGAR CSV
    # ------------------------
    df = pd.read_csv(args.csv)

    # Si hay columna de rating → crear sentimiento
    if args.sentiment is not None:
        df['sentiment_mapped'] = df[args.sentiment].apply(map_sentiment)

    # ------------------------
    # MODO 1: PREDICT
    # ------------------------
    if args.mode == "predict":

        df_sample = df.sample(n=min(args.samples, len(df)))

        if args.shot == "0":
            template = """You are a sentiment classifier. Calssify the sentiment of the following text. 
            Respond with ONLY ONE word: 'Positive', 'Negative' or 'Neutral'. Text: "In my expierence, a lot of matches never reply. 
            It takes away form the overall experience."

            Text: {text}

            Sentiment:"""
        elif args.shot == "1":
            template = """You are a sentiment classifier. Calssify the sentiment of the following text. 
            Respond with ONLY ONE word: 'Positive', 'Negative' or 'Neutral'. Text: "In my expierence, a lot of matches never reply. 
            It takes away form the overall experience."
            
            Examples: {examples}

            Text: {text}

            Sentiment:"""
        elif args.shot == "few":
            template = """You are a sentiment classifier. Calssify the sentiment of the following text. 
            Respond with ONLY ONE word: 'Positive', 'Negative' or 'Neutral'. Text: "In my expierence, a lot of matches never reply. 
            It takes away form the overall experience."
            
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

        print(f"Predicciones guardadas en {output_file}")

    # ------------------------
    # MODO 2: OVERSAMPLING
    # ------------------------
    elif args.mode == "oversample":

        if args.sentiment is None:
            raise ValueError("Necesitas --sentiment con valores 1-5")

        template = """You are generating synthetic data for sentiment classification.
            Generate a paraphrased version of the following text.
            Keep the SAME sentiment: {label}

            Text: {text}

            New text:"""

        prompt = PromptTemplate.from_template(template)
        chain = prompt | model

        new_rows = []

        df_sample = df.sample(n=min(args.samples, len(df)))

        for _, row in df_sample.iterrows():
            text = str(row[args.target])
            label = row['sentiment_mapped']

            try:
                new_text = chain.invoke({'text': text, 'label': label}).strip()
            except:
                continue

            new_rows.append({
                args.target: new_text,
                'sentiment_mapped': label
            })

        df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        output_file = "oversampled.csv"
        df_aug.to_csv(output_file, index=False)

        print(f"Dataset aumentado guardado en {output_file}")


# ------------------------
# EJECUCIÓN
# ------------------------
if __name__ == "__main__":
    main()
