from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# =========================
# CORS (ESSENCIAL pro frontend)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois pode restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Caminho base
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
# Carregar scaler
# =========================
try:
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

# =========================
# Entrada
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
# Saída (IMPORTANTE)
# =========================
class PredictionOutput(BaseModel):
    resultado: int
    diagnostico: str

# =========================
# Home
# =========================
@app.get("/")
def home():
    return {"mensagem": "API de Diabetes rodando 🚀"}

# =========================
# Predição
# =========================
@app.post("/predict", response_model=PredictionOutput)
def predict(data: DiabetesInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    try:
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

        if scaler is not None:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        return {
            "resultado": int(prediction),
            "diagnostico": "Diabetes" if prediction == 1 else "Sem diabetes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))