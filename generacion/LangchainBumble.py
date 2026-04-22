import argparse
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

# ------------------------
# ARGUMENTOS
# ------------------------
parser = argparse.ArgumentParser(description="Sentiment classification with Ollama")

parser.add_argument('--mode', type=str, required=True, choices=['predict', 'oversample'])
parser.add_argument('--model', type=str, default='gemma2:2b')

# CSV settings
parser.add_argument('--csv', type=str, help='Path al CSV')
parser.add_argument('--text_column', type=str, help='Columna de texto')
parser.add_argument('--label_column', type=str, default=None, help='Columna de etiqueta (para oversample)')
parser.add_argument('--n_samples', type=int, default=20)

args = parser.parse_args()

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
# MODO 1: PREDICCIÓN
# ------------------------
if args.mode == "predict":

    df = pd.read_csv(args.csv)

    # seleccionar muestras aleatorias
    df_sample = df.sample(n=min(args.n_samples, len(df)))

    template = """Classify the sentiment of the following text.
        Respond ONLY with one word: Positive, Negative or Neutral.

        Text: {text}
        Answer:"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | model

    predictions = []

    for text in df_sample[args.text_column]:
        try:
            ans = chain.invoke({'text': str(text)}).strip()
        except:
            ans = "ERROR"

        # limpiar salida
        if ans not in ["Positive", "Negative", "Neutral"]:
            ans = "Neutral"

        predictions.append(ans)

    df_sample['prediction'] = predictions

    output_file = "predictions.csv"
    df_sample.to_csv(output_file, index=False)

    print(f"Predicciones guardadas en {output_file}")


# ------------------------
# MODO 2: AUGMENTACIÓN
# ------------------------
elif args.mode == "oversample":

    df = pd.read_csv(args.csv)

    if args.label_column is None:
        raise ValueError("Necesitas --label_column para oversample")

    template = """You are generating synthetic data for sentiment classification.

Generate a paraphrased version of the following text.
Keep the SAME sentiment: {label}
s
Text: {text}

New text:"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | model

    new_rows = []

    df_sample = df.sample(n=min(args.n_samples, len(df)))

    for _, row in df_sample.iterrows():
        text = str(row[args.text_column])
        label = row[args.label_column]

        try:
            new_text = chain.invoke({'text': text, 'label': label}).strip()
        except:
            continue

        new_rows.append({
            args.text_column: new_text,
            args.label_column: label
        })

    df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    output_file = "oversampled.csv"
    df_aug.to_csv(output_file, index=False)

    print(f"Dataset aumentado guardado en {output_file}")
