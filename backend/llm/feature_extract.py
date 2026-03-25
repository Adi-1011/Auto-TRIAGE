from google import genai
import joblib
import numpy as np
import pandas as pd
import json
import os

gem3 = "gemini-3-flash-preview"
client = genai.Client(api_key=os.getenv("GEMENI_API_KEY"))
querry = input("type your querry here")
querry += """

Based on the above and any image provided, estimate the most clinically plausible values for the following 9 features based on the described symptoms. Do NOT return null for any field — always provide a reasonable estimated value based on the symptom description. Return ONLY a valid JSON object with no extra text, no markdown, no explanation:
{
    "age": <number>,
    "heart_rate": <estimated number in bpm>,
    "respiratory_rate": <estimated number in breaths per min>,
    "systolic_bp": <estimated number in mmHg>,
    "spo2": <estimated percentage 75-100>,
    "temprature": <estimated number in celsius>,
    "wbc": <estimated number>,
    "crp": <estimated number>,
    "comorbidty": <0 or 1>
}
"""
file = client.files.upload(file="data/Images/condition_1.jpg")
response = client.models.generate_content(
    model=gem3,contents=[querry, file]
)
print()
print(response.text)

raw = response.text.strip()

if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]

features_json = json.loads(raw)

model = joblib.load("models/saved/xgb/xgb_model.pkl")
le = joblib.load("models/saved/xgb/label_encoder.pkl")

def classify_patient(feature_json: dict):

    column_order = [
        "age",
        "heart_rate", 
        "respiratory_rate",
        "systolic_bp",
        "spo2",
        "temprature",   
        "wbc",
        "crp",
        "comorbidty"    
    ]
    df = pd.DataFrame([feature_json])[column_order]
    df = df.astype(float)

    prediction = model.predict(df)
    prediction_label = le.inverse_transform(prediction)[0]

    return prediction_label


result = classify_patient(features_json)
print("\nTRIAGE CLASSIFICATION RESULT\n")
print(f"patient symptoms as input features: {features_json}")
print(f"Patient classification: {result}")