from fastapi import FastAPI
from pydantic import BaseModel
from backend.llm.feature_extract import classify_patient, feature_extract_from_text
from fastapi.middleware.cors import CORSMiddleware


class queryInput(BaseModel):
    text:str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post('/predict')
def predict(data: queryInput):
    features = feature_extract_from_text(data.text)
    result = classify_patient(features)
    return {
        "features": features,
        "prediction": result
    }

