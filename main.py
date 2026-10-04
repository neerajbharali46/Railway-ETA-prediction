from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://127.0.0.1:5501",
    "https://railway-eta-frontend.onrender.com"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df=pd.read_csv("augmented_train_eta_dataset.csv")
df["Train No."] = (
    df["Train No."]
    .astype(str)
    .str.strip()
    .str.strip("'\"")
)

Model=joblib.load("linear_regression_eta.pkl")

class TrainRequest(BaseModel):
    train_no:str
    current_station: str
@app.get("/")
def home():
    return {"message": "Indian Railway ETA Prediction API is running!"}

@app.post("/predict")
def predict_train(data: TrainRequest):
    train_no = data.train_no.strip()
    current_station = data.current_station.strip().upper()

    train_data = df[
    (df["Train No."].astype(str).str.strip() == train_no.strip()) &
    (df["Station Name"].astype(str).str.strip().str.upper() == current_station.strip().upper())
    ]

    if train_data.empty:
        return {"error": "Train or station not found"}

    row = train_data.iloc[0]

    features = pd.DataFrame([{
        "Current Delay (min)": row["Current Delay (min)"],
        "Current Speed (km/h)": row["Current Speed (km/h)"],
        "Distance to Next Station (km)": row["Distance to Next Station (km)"],
        "Historical Avg Delay (min)": row["Historical Avg Delay (min)"],
        "Congestion": row["Congestion"],
        "Weather": row["Weather"]
    }])

    predicted_delay = Model.predict(features)[0]
    arrival_time = str(row["Arrival time"]).strip().strip("'\"")
    scheduled_arrival = datetime.strptime(arrival_time, "%H:%M:%S")
    predicted_eta = scheduled_arrival + timedelta(minutes=float(predicted_delay))


    return {
    "train_no": train_no,
    "current_station": current_station,
    "scheduled_arrival": scheduled_arrival.strftime("%H:%M"),
    "current_delay": float(row["Current Delay (min)"]),
    "current_speed": float(row["Current Speed (km/h)"]),
    "distance_to_next_station": float(row["Distance to Next Station (km)"]),
    "historical_avg_delay": float(row["Historical Avg Delay (min)"]),
    "congestion": row["Congestion"],
    "weather": row["Weather"],
    "predicted_future_delay": round(float(predicted_delay), 2),
    "predicted_eta": predicted_eta.strftime("%H:%M")
}
