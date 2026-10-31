from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="API Predicción Educativa")

# Cargar el modelo al iniciar
MODEL_PATH = "modelo_educativo.joblib"
if os.path.exists(MODEL_PATH):
    artefacto = joblib.load(MODEL_PATH)
    modelo = artefacto['modelo']
    predictoras = artefacto['columnas_numericas'] + artefacto['columnas_categoricas']
else:
    raise Exception("Modelo no encontrado")

# Definir el esquema de datos que recibirá la API
class Estudiante(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int

@app.post("/predecir")
async def predecir(estudiante: Estudiante):
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([estudiante.dict()])
        df = df[predictoras] # Asegurar el orden correcto
        
        # Predecir
        prediccion = modelo.predict(df)[0]
        probabilidades = modelo.predict_proba(df)[0]
        dict_prob = dict(zip(modelo.classes_, probabilidades))
        
        return {
            "estado": "éxito",
            "prediccion": prediccion,
            "probabilidades": dict_prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))