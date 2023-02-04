from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd



app = FastAPI()


class ScoreItem(BaseModel):
    Pregnancies:int
    Glucose: int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int

with open("Log_model.pkl","rb") as f:
    model = pickle.load(f)



@app.post("/predict")
async def getData(item:ScoreItem):
    data = pd.DataFrame([item.dict().values()],columns= item.dict().keys())
    y = model.predict(data)

    return {'prediction':int(y)}


@app.get("/")
async def get():
    return {"hello":"world"}