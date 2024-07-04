import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)

# variables
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
dict_res = {0: 'Not-Diabetes', 1: 'Diabetes'}
model_path = './models/pipeline.pkl'

# app instance
app = FastAPI()

# Global variable for the model
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained model from a file."""
    global model
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError as e:
        logging.error(f"Model not found: {e}")
        # Consider adding code to stop the server or handle this error appropriately

class DataInput(BaseModel):
    data: list

@app.get("/")
async def root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict")
async def predict(input_data: DataInput):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded")
        df = pd.DataFrame(input_data.data, columns=columns)
        predictions = model.predict(df)
        results = [dict_res[pred] for pred in predictions]
    
        return {"predictions": results}
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)