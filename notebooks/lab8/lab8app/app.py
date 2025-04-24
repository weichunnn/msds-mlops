import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

iris_class_mapping = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

app = FastAPI()

try:
    model = mlflow.pyfunc.load_model('./model')
except Exception as e:
    print(f"Error loading model from path: {e}")
    model = None

@app.get("/")
def read_root():
    return {"message": "Iris classifier API is running!"}

@app.post("/predict")
def predict_iris(features: IrisFeatures):
    if model is None:
        return {"error": "Model not loaded. Check logs."}
    try:
        data_df = pd.DataFrame([features.model_dump()])
        pred_int = model.predict(data_df)
        prediction = iris_class_mapping.get(pred_int[0], "Unknown class")
        return {"prediction": prediction}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
