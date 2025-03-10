import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Charger le modèle entraîné
@st.cache_resource
def load_model():
    with open("credit_scoring_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Titre de l'application
st.title("🧮 Crédit Scoring - Prédiction de Risque")

# Formulaire pour entrer les données du client
st.sidebar.header("📋 Informations Client")
age = st.sidebar.slider("Âge", 18, 80, 30)
revenu = st.sidebar.number_input("Revenu mensuel (en $)", min_value=0, value=3000, step=500)
dette = st.sidebar.number_input("Dette actuelle (en $)", min_value=0, value=5000, step=1000)
historique_credit = st.sidebar.selectbox("Historique de crédit", ["Mauvais", "Moyen", "Bon"])
nombre_cartes = st.sidebar.slider("Nombre de cartes de crédit", 0, 10, 2)

# Transformation des données
historique_map = {"Mauvais": 0, "Moyen": 1, "Bon": 2}
features = np.array([[age, revenu, dette, historique_map[historique_credit], nombre_cartes]])

# Prédiction du modèle
if st.sidebar.button("📊 Prédire le score"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    # Affichage des résultats
    st.subheader("📌 Résultat de la Prédiction")
    if prediction == 1:
        st.success(f"✅ Crédit approuvé avec une probabilité de {proba[1]*100:.2f}%")
    else:
        st.error(f"❌ Crédit refusé avec une probabilité de {proba[0]*100:.2f}%")

    # Graphique des probabilités
    st.subheader("📊 Probabilités du Modèle")
    df_proba = pd.DataFrame({"Classe": ["Refusé", "Approuvé"], "Probabilité": proba})
    st.bar_chart(df_proba.set_index("Classe"))

