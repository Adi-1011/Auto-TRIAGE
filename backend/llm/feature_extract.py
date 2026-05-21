import base64
import json
import os
import tempfile

import joblib
import pandas as pd
from google import genai


gem3 = "gemini-3-flash-preview"
api_key = os.getenv("GEMENI_API_KEY") or os.getenv("GEMENI_API_KEY")
client = genai.Client(api_key=api_key)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLUMNS = [
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

FEATURE_DEFAULTS = {
    "age": 40,
    "hr": 80,
    "rr": 16,
    "systolic_bp": 115,
    "spo2": 97,
    "temp": 37,
    "wbc_count": 7.5,
    "delta_hr": 0,
    "delta_rr": 0,
    "delta_spo2": 0,
    "delta_systolic_bp": 0,
    "delta_temp": 0,
    "comorbidity": 0,
    "crp": 10,
}


def _clean_json_response(raw: str) -> dict:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def _image_data_url_to_temp_file(image_data_url: str | None) -> str | None:
    if not image_data_url:
        return None

    header, _, payload = image_data_url.partition(",")
    suffix = ".jpg"

    if "png" in header:
        suffix = ".png"
    elif "webp" in header:
        suffix = ".webp"
    elif "jpeg" in header or "jpg" in header:
        suffix = ".jpg"

    image_bytes = base64.b64decode(payload or image_data_url)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(image_bytes)
    tmp.close()
    return tmp.name


def _coerce_features(features_json: dict) -> dict:
    aliases = {
        "heart_rate": "hr",
        "respiratory_rate": "rr",
        "temprature": "temp",
        "temperature": "temp",
        "wbc": "wbc_count",
        "comorbidty": "comorbidity",
    }

    normalized = {
        aliases.get(key, key): value
        for key, value in features_json.items()
    }

    for column in FEATURE_COLUMNS:
        if column not in normalized or normalized[column] is None:
            normalized[column] = FEATURE_DEFAULTS[column]

    normalized["comorbidity"] = int(round(float(normalized["comorbidity"])))
    normalized["comorbidity"] = max(0, min(1, normalized["comorbidity"]))

    for column in FEATURE_COLUMNS:
        normalized[column] = float(normalized[column])

    return {column: normalized[column] for column in FEATURE_COLUMNS}


def feature_extract_from_text(query: str, image_data_url: str | None = None) -> dict:
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GEMENI_API_KEY.")

    prompt = f"""
    Patient description:
    {query}

    Based on the symptom description and any uploaded image provided, estimate the most clinically plausible values for the following 14 features used by the latest Auto-TRIAGE model.

    Rules:
    - Do not return null for any field.
    - Return a reasonable numeric estimate for every field.
    - Use the uploaded image only as supporting evidence for visible symptoms such as wounds, rash, swelling, pallor, cyanosis, injury, or distress.
    - Temporal delta fields describe short-term change. Use positive values when the description suggests worsening/rising values, negative values when it suggests dropping values, and 0 when no time trend is described.
    - Return only a valid JSON object. Do not return markdown, comments, or explanation.

    Required JSON schema:
    {{
        "age": <number>,
        "hr": <estimated heart rate in bpm>,
        "rr": <estimated respiratory rate in breaths per min>,
        "systolic_bp": <estimated number in mmHg>,
        "spo2": <estimated percentage 75-100>,
        "temp": <estimated body temperature in celsius>,
        "wbc_count": <estimated WBC count>,
        "delta_hr": <short-term heart rate change>,
        "delta_rr": <short-term respiratory rate change>,
        "delta_spo2": <short-term SpO2 change>,
        "delta_systolic_bp": <short-term systolic BP change>,
        "delta_temp": <short-term temperature change>,
        "comorbidity": <0 or 1>,
        "crp": <estimated CRP level>
    }}
    """

    image_path = _image_data_url_to_temp_file(image_data_url)

    try:
        contents = [prompt]

        if image_path:
            uploaded_file = client.files.upload(file=image_path)
            contents.append(uploaded_file)

        response = client.models.generate_content(
            model=gem3,
            contents=contents,
        )

        features_json = _clean_json_response(response.text)
        return _coerce_features(features_json)
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


model_path = os.path.join(BASE_DIR, "../../models/saved3/xgb/xgb_model.pkl")
le_path = os.path.join(BASE_DIR, "../../models/saved3/xgb/label_encoder.pkl")
model = joblib.load(model_path)
le = joblib.load(le_path)


def classify_patient(feature_json: dict) -> str:
    features = _coerce_features(feature_json)
    column_order = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    df = pd.DataFrame([features])[column_order]
    df = df.astype(float)

    prediction = model.predict(df)
    prediction_label = le.inverse_transform(prediction)[0]

    return prediction_label
