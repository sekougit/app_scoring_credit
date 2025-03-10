import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 📌 1️⃣ Fonction pour générer des données factices
def generate_data(n_samples=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        "age": np.random.randint(18, 80, n_samples),
        "revenu": np.random.randint(1000, 10000, n_samples),
        "dette": np.random.randint(0, 15000, n_samples),
        "historique_credit": np.random.choice(["Mauvais", "Moyen", "Bon"], n_samples),
        "nombre_cartes": np.random.randint(0, 10, n_samples),
        "credit_accepte": np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # 0 = refusé, 1 = accepté
    })

    # Encodage de la variable catégorielle "historique_credit"
    historique_map = {"Mauvais": 0, "Moyen": 1, "Bon": 2}
    data["historique_credit"] = data["historique_credit"].map(historique_map)

    return data

# 📌 2️⃣ Fonction pour entraîner le modèle
def train_model():
    data = generate_data()
    X = data.drop(columns=["credit_accepte"])
    y = data["credit_accepte"]

    # Division en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluation du modèle
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy du modèle : {accuracy * 100:.2f}%")

    return model

# 📌 3️⃣ Fonction pour récupérer un modèle déjà entraîné
def get_trained_model():
    return train_model()  # Renvoie directement un modèle entraîné
