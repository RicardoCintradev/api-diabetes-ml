from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
import traceback

app = FastAPI()

# =========================
# Caminho base do projeto
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# Carregar modelo
# =========================
try:
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Erro ao carregar model.pkl:", e)
    model = None

# =========================
# Carregar scaler (opcional)
# =========================
try:
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None  # se não tiver scaler, continua sem ele

# =========================
# Schema de entrada (validação)
# =========================
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

# =========================
# Rota inicial
# =========================
@app.get("/")
def home():
    return {"mensagem": "API de Diabetes rodando 🚀"}

# =========================
# Predição
# =========================
@app.post("/predict")
def predict(data: DiabetesInput):
    try:
        if model is None:
            return {"erro": "Modelo não carregado"}

        values = [
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]

        input_data = np.array(values).reshape(1, -1)

        # aplica scaler se existir
        if scaler is not None:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        return {
            "resultado": int(prediction),
            "diagnostico": "Diabetes" if prediction == 1 else "Sem diabetes"
        }

    except Exception as e:
        return {
            "erro": str(e),
            "trace": traceback.format_exc()
        }