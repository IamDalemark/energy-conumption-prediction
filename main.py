from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
from preprocess import preprocessDataflow


class InputData(BaseModel):
    building_type: str
    square_footage: int
    number_of_occupants: int
    appliances_used: int

with open("EnergyConsumptionLinearModel.pkl", "rb") as f:
    model = pickle.load(f) 
with open("EnergyConsumptionScaler.pkl", "rb") as f: 
    scaler = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    print("get")
    return {"message": "Energy consumption API is working"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([{
        "BuildingType": data.building_type,
        "SquareFootage": data.square_footage,
        "NumberOfOccupants": data.number_of_occupants,
        "AppliancesUsed": data.appliances_used
    }])
    processed = preprocessDataflow(df)
    scaled_features = scaler.transform(processed)
    prediction = model.predict(scaled_features)[0]
    
    coefficients = model.coef_
    contributions = scaled_features[0] * coefficients
    factors = {
        "building_type": contributions[0],
        "square_footage": contributions[1],
        "number_of_occupants": contributions[2],
        "appliances_used": contributions[3]
    }
    return {"energy_consumption": prediction, "unit": "kpwh", "factors": factors}

@app.get("/dataset")
def dataset(page=1, limit=50):
    page = int(page)
    limit = int(limit)
    df = pd.read_csv("train_energy_data.csv")
    df = df.drop(columns=["DayOfWeek", "AverageTemperature"], axis = 1 )
    df.rename(columns={
    "BuildingType": "building_type",
    "SquareFootage": "square_footage",
    "NumberOfOccupants": "number_of_occupants",
    "AppliancesUsed": "appliances_used",
    "EnergyConsumption": "energy_consumption"
}, inplace=True)
    total = len(df)
    pages = (total + limit - 1) // limit 
    start = (page - 1) * limit
    end = start + limit
    data = df.iloc[start:end].to_dict(orient="records")
    return {
        "page": page,
        "limit": limit,
        "total": len(df),
        "data": data,
        "pages": pages
    }
 