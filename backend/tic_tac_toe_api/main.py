# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Cargar modelos (debe existir carpeta models con los .pkl)
MODEL_DIR = "models"
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
gb = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
mlp = joblib.load(os.path.join(MODEL_DIR, "mlp.pkl"))
scaler_mlp = joblib.load(os.path.join(MODEL_DIR, "scaler_mlp.pkl"))

mapping_tablero = {"X": 1, "x": 1, "O": -1, "o": -1, "_": 0, "b": 0}
mapping_turno = {"X": 1, "x": 1, "O": -1, "o": -1}

app = FastAPI(title="TicTacToe AI API")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto por ["http://localhost:3000"] si solo permites tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    board: list   # lista de 9 elementos: "X","O","_" ...
    turn: str     # "X" o "O"
    ai: int       # 1 = RF, 2 = GB, 3 = MLP

def encode_input(board, turn):
    if not isinstance(board, (list, tuple)) or len(board) != 9:
        raise ValueError("board debe ser lista de 9 elementos")
    t_num = [mapping_tablero.get(str(x), 0) for x in board]
    s_num = mapping_turno.get(turn, 1)
    return np.array(t_num + [s_num]).reshape(1, -1)

@app.post("/predict")
def predict(data: InputData):
    try:
        X = encode_input(data.board, data.turn)
    except ValueError as e:
        return {"error": str(e)}

    if data.ai == 1:
        pred = int(rf.predict(X)[0])
    elif data.ai == 2:
        pred = int(gb.predict(X)[0])
    elif data.ai == 3:
        Xs = scaler_mlp.transform(X)
        pred = int(mlp.predict(Xs)[0])
    else:
        return {"error": "ai debe ser 1, 2 o 3"}

    return {"best_move": pred}
# .venv\Scripts\activate.bat
# uvicorn main:app --reload