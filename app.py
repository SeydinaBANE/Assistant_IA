# app.py - Application Streamlit pour Data Whisperer (version open-source)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

st.set_page_config(page_title="Décodeur de Stats", layout="wide")
st.title("🤖 Décodeur de Stats – Analysez vos données avec un LLM open-source")

@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Peut être remplacé par mistralai/Mistral-7B-Instruct-v0.2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

pipe = load_model()

uploaded_file = st.file_uploader("📁 Importez votre fichier CSV", type=["csv"])
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Aperçu du jeu de données")
        st.dataframe(df.head())

        st.subheader("🧠 Posez une question sur vos données")
        user_question = st.text_input("Ex : Quelle colonne contient le plus de valeurs manquantes ?")

        if user_question:
            prompt = f"""
            Tu es un assistant expert en data science. Voici un DataFrame :

            Aperçu :
            {df.head().to_string()}

            Types des colonnes :
            {df.dtypes.to_string()}

            Question :
            {user_question}

            Fournis uniquement le code Python (pandas + matplotlib si nécessaire), sans commentaires ni explication.
            """

            try:
                outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.3)
                code = outputs[0]['generated_text'].split("\n", 1)[-1]  # Nettoyage du prompt original si reproduit

                st.subheader("📝 Code généré")
                st.code(code, language="python")

                st.subheader("📈 Résultat")
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        local_vars = {"df": df, "plt": plt, "pd": pd}
                        exec(code, {}, local_vars)

                    if "plt" in code:
                        st.pyplot(plt)
                except Exception as e:
                    st.error(f"⚠️ Erreur lors de l'exécution du code : {e}")

            except Exception as e:
                st.error(f"⚠️ Erreur lors de la génération du code avec le modèle : {e}")

    except Exception as e:
        st.error(f"⚠️ Échec de lecture du fichier CSV : {e}")

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")
