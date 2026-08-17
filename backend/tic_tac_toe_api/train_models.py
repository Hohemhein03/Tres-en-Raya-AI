# train_models.py
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("models", exist_ok=True)

# URL dataset (usa el dataset que mencionaste)
url = "https://raw.githubusercontent.com/Hohemhein03/Dataset-Tic-Tac-Toe/refs/heads/main/dataset.csv"
raw = requests.get(url).text
raw_clean = raw.replace('"', '')
df = pd.read_csv(StringIO(raw_clean), header=None)

# Columnas
df.columns = [f"pos_{i}" for i in range(9)] + ["turn", "best_move"]

# Mapeos
mapping_tablero = {"X": 1, "x": 1, "O": -1, "o": -1, "_": 0, "b": 0}
mapping_turno = {"X": 1, "x": 1, "O": -1, "o": -1}

for col in [f"pos_{i}" for i in range(9)]:
    df[col] = df[col].map(mapping_tablero).fillna(0).astype(int)

df["turn"] = df["turn"].map(mapping_turno).fillna(1).astype(int)
df["best_move"] = df["best_move"].astype(int)

X = df.drop(columns=["best_move"])
y = df["best_move"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RANDOM FOREST
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")

# GRADIENT BOOSTING
gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
joblib.dump(gb, "models/gradient_boosting.pkl")

# MLP (red neuronal) -- requiere escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, activation="relu", random_state=42)
mlp.fit(X_train_scaled, y_train)

joblib.dump(mlp, "models/mlp.pkl")
joblib.dump(scaler, "models/scaler_mlp.pkl")

print("Modelos guardados en la carpeta models/:")
print(" - models/random_forest.pkl")
print(" - models/gradient_boosting.pkl")
print(" - models/mlp.pkl")
print(" - models/scaler_mlp.pkl")
