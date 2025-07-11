from fastapi import FastAPI, Request
from typing import List
import pandas as pd
from app.scoring import calculate_rfm
from app.models import Segmenter
from app.churn import ChurnPredictor
import os
import joblib

app = FastAPI()

# Load RFM segmenter
segmenter = Segmenter()
if os.path.exists("model/cluster_model.pkl"):
    segmenter.load("model/cluster_model.pkl")
else:
    print("⚠️ cluster_model.pkl not found. Please train the model.")

# Load churn model
churn_model = ChurnPredictor()
if os.path.exists("model/churn_model.pkl"):
    churn_model.load("model/churn_model.pkl")
else:
    print("⚠️ churn_model.pkl not found. Please train the model.")

@app.post("/score")
def score(data: List[dict]):
    try:
        df = pd.DataFrame(data)
        print("✅ Received Data:")
        print(df.head())
        print("🧾 Columns:", df.columns.tolist())
        rfm = calculate_rfm(df)
        return rfm.reset_index().to_dict(orient="records")
    except Exception as e:
        return {"error": f"Score calculation failed: {str(e)}"}


@app.post("/segment")
def segment(data: List[dict]):
    try:
        df = pd.DataFrame(data)
        rfm = calculate_rfm(df)
        rfm_values = rfm[['Recency', 'Frequency', 'Monetary']]
        segments = segmenter.model.predict(rfm_values)
        rfm['Segment'] = segments
        return rfm.reset_index().to_dict(orient="records")
    except Exception as e:
        return {"error": f"Segmentation failed: {str(e)}"}

@app.post("/churn")
def churn(data: List[dict]):
    try:
        df = pd.DataFrame(data)
        probs = churn_model.predict(df[['Recency']])
        df['churn_probability'] = probs
        if 'CustomerID' in df.columns:
            return df[['CustomerID', 'churn_probability']].to_dict(orient="records")
        else:
            return df[['churn_probability']].to_dict(orient="records")
    except Exception as e:
        return {"error": f"Churn prediction failed: {str(e)}"}

