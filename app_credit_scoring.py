import streamlit as st
from models import get_trained_model  # Importer le modèle entraîné

# Charger le modèle
st.title("Application de Scoring de Crédit")

model = get_trained_model()  # Utilise le modèle entraîné directement

# Interface utilisateur pour entrer les données
age = st.number_input("Âge", min_value=18, max_value=80, value=30)
revenu = st.number_input("Revenu mensuel", min_value=1000, max_value=10000, value=3000)
dette = st.number_input("Dette totale", min_value=0, max_value=15000, value=2000)
historique_credit = st.selectbox("Historique de crédit", ["Mauvais", "Moyen", "Bon"])
nombre_cartes = st.number_input("Nombre de cartes de crédit", min_value=0, max_value=10, value=2)

# Encodage de la variable historique de crédit
historique_map = {"Mauvais": 0, "Moyen": 1, "Bon": 2}
historique_credit = historique_map[historique_credit]

# Prédiction
if st.button("Prédire l'acceptation du crédit"):
    input_data = [[age, revenu, dette, historique_credit, nombre_cartes]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Crédit accepté !")
    else:
        st.error("❌ Crédit refusé.")
