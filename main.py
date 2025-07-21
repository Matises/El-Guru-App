from fastapi import FastAPI
import joblib

app = FastAPI()
modelo = joblib.load("modelo.pkl")

@app.get("/")
def home():
    return {"mensaje": "El Gurú está activo"}

@app.post("/predecir/")
def predecir(datos: dict):
    entrada = [list(datos.values())]
    prediccion = modelo.predict(entrada)
    return {"predicción": int(prediccion[0])}
