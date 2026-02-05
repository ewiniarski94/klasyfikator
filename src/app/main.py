from fastapi import FastAPI
import joblib
from schemas import IrisData

app = FastAPI()
model = joblib.load("iris.joblib")

@app.post("/predict")
def predict(data: IrisData):
    input_data = ([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
