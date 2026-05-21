import joblib
import pandas as pd

model = joblib.load("models/saved3/xgb/xgb_model.pkl")
le = joblib.load("models/saved3/xgb/label_encoder.pkl")

columns = [
    "age",
    "hr",
    "rr",
    "systolic_bp",
    "spo2",
    "temp",
    "wbc_count",
    "delta_hr",
    "delta_rr",
    "delta_spo2",
    "delta_systolic_bp",
    "delta_temp",
    "comorbidity",
    "crp",
]

patient = {
    "age": 67,
    "hr": 125,
    "rr": 34,
    "systolic_bp": 82,
    "spo2": 84,
    "temp": 39.5,
    "wbc_count": 18,
    "delta_hr": 15,
    "delta_rr": 7,
    "delta_spo2": -8,
    "delta_systolic_bp": -15,
    "delta_temp": 1.2,
    "comorbidity": 1,
    "crp": 120,
}

df = pd.DataFrame([patient])[columns]

pred = model.predict(df)
label = le.inverse_transform(pred)[0]
print(df)
print("Prediction:", label)