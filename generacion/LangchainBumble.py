import argparse
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os

# ------------------------
# MAIN
# ------------------------
def main():

    parser = argparse.ArgumentParser(description="Sentiment classification with Ollama")

    parser.add_argument('--mode', type=str, required=True, choices=['predict', 'oversample'])
    parser.add_argument('--model', type=str, default='granite4:350m-h')

    # CSV
    parser.add_argument('--csv', type=str)
    parser.add_argument('--target', type=str, help='Columna de comentarios')
    parser.add_argument('--sentiment', type=str, default=None, help='Columna numérica 1-5')
    parser.add_argument('--score', type=str, default=None, help='Score de la opinión a generar en oversampling')
    parser.add_argument('--samples', type=int, default=10, help='Número de instancias a calsificar')
    parser.add_argument('--prompt', type=str, required=True, help='Archivo .txt con el prompt')

    args = parser.parse_args()

    model=load_model(args.mode, args.model)

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
        
        if args.prompt is None:
            raise ValueError("You must provide --prompt (txt) including the prompt")
        
        df = load_csv(args.csv, args.sentiment)

        if args.target not in df.columns:
            raise ValueError(f"Column '{args.target}' not found in CSV")

        if args.sentiment not in df.columns:
            raise ValueError(f"Column '{args.sentiment}' not found in CSV")
        
        df_sample = df.sample(n=min(args.samples, len(df)))

        #Abrir el prompt txt
        with open(args.prompt, "r", encoding="utf-8") as f:
            promptxt = f.read()

        template = promptxt + """
            Text: {text}

            Your response (one word only):"""
     
        prompt = PromptTemplate.from_template(template)
        chain = prompt | model

        predictions = []

        for text in df_sample[args.target]:
            try:
                raw_ans = chain.invoke({'text': str(text)}).strip()
                ans_lower = raw_ans.lower() # Pasamos a minúsculas para evitar fallos
                
                # Buscamos la palabra clave dentro de la respuesta
                if "positive" in ans_lower:
                    ans = "Positive"
                elif "negative" in ans_lower:
                    ans = "Negative"
                elif "neutral" in ans_lower:
                    ans = "Neutral"
                else:
                    ans = "NotExpected"
                    print(f"[!] Respuesta rara del modelo: {raw_ans}")
                    
            except Exception as e: # <--- AHORA ATRAPAMOS EL ERROR REAL
                ans = "NotExpected"
                print(f"\n[ERROR FATAL] Fallo al llamar a Ollama: {e}") # <--- LO IMPRIMIMOS

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


        ##------ CSV ------

        COLUMNS = ["modelo", "prompt", "entrada", "salida", "real"]
        output_file = "predictions_generative.csv"

        if os.path.exists(output_file):
            df_existing = pd.read_csv(output_file)
        else:
            df_existing = pd.DataFrame(columns=COLUMNS)

        new_rows = []

        for idx, row in df_sample.iterrows():
            new_rows.append({
                "modelo": args.model,
                "prompt": promptxt,
                "entrada": row[args.target],
                "salida": row['prediction'],
                "real": row['sentiment_mapped'] 
            })

        df_new = pd.DataFrame(new_rows, columns=COLUMNS)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        df_updated.to_csv(output_file, index=False)

        print(f"Predictions saved in {output_file} (total rows: {len(df_updated)}, added: {len(new_rows)})")
        df_sample.to_csv("predictions_detailed.csv", index=False)
        print(f"Detailed predictions saved in predictions_detailed.csv")



    # ------------------------
    # MODO 2: OVERSAMPLING
    # ------------------------
    elif args.mode == "oversample":

        if args.score is None:
            raise ValueError("You must provide --score (1-5)")
        
        if args.prompt is None:
            raise ValueError("You must provide --prompt (txt) including the prompt")
        
        # Mapear el número a etiqueta
        sentiment_label = map_sentiment(args.score)

        #Abrir el prompt txt
        with open(args.prompt, "r", encoding="utf-8") as f:
            promptxt = f.read()

        prompt = PromptTemplate.from_template(promptxt)
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

        prompt_name = os.path.splitext(os.path.basename(args.prompt))[0]
        output_file = f"{prompt_name}_{sentiment_label}.csv"
        df_new.to_csv(output_file, index=False)

        print(f"Generated {len(new_rows)} new samples")
        print(f"New dataset saved to {output_file}")


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
            temperature=0,        # creatividad/aleatoriedad del modelo (0: estricto)
            num_predict=20,       # número máximo de tokens  
            repeat_penalty=1.1,   # penalización por repetir palabras o frases
            top_k=10,             # tamaño del vocabulario candidato
            top_p=0.5             # masa de probabilidad acumulada
        )
    else:
        model = OllamaLLM(
            model=name,  
            temperature=0.35,     # creatividad/aleatoriedad del modelo (1: creativo)
            num_predict=45,       # número máximo de tokens  
            repeat_penalty=1.4,  # penalización por repetir palabras o frases 
            top_k=10,             # tamaño del vocabulario candidato
            top_p=0.5             # masa de probabilidad acumulada
        )

    return model


# ------------------------
# EJECUCIÓN
# ------------------------
if __name__ == "__main__":
    main()
