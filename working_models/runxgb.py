import json, joblib, pandas as pd
from tabulate import tabulate
import os 

def export_html_report(results, filename="triage_report.html"):
    report_dir = r"C:\Users\Aditya\Documents\FYRP DATA\MAIN BUILD\Auto-TRIAGE\working_models\classification report"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    filepath = os.path.join(report_dir, filename)
    # Convert list to DataFrame 
    df = pd.DataFrame(results, columns=["Patient ID", "Predicted Class", "Confidence (%)"])
    # Custom HTML
    html_template = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f7f6; padding: 40px; }}
            .card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h2 {{ color: #2c3e50; border-bottom: 3px solid #3498db; display: inline-block; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f1f1f1; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Auto-Triage Final Summary</h2>
            {df.to_html(index=False, classes='table')}
        </div>
    </body>
    </html>
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\nHTML Report exported successfully: {filepath}")
    import webbrowser
    webbrowser.open(filepath)

def main():
    # Load model, demodata etc...
    model = joblib.load("models/saved3/xgb/xgb_model.pkl")
    enc = joblib.load("models/saved3/xgb/label_encoder.pkl")
    
    with open("working_models/sample_records.json", "r") as f:
        patients = json.load(f)

    expected_features = [
        'age', 'hr', 'rr', 'systolic_bp', 'spo2', 'temp', 'wbc_count', 
        'delta_hr', 'delta_rr', 'delta_spo2', 'delta_systolic_bp', 'delta_temp', 
        'comorbidity', 'crp'
    ]

    results = []
    
    for p in patients:

        df = pd.DataFrame([p]).drop(columns=["patient_id"])
        
        # Check if all features exist to prevent crashing
        try:
            df = df[expected_features]
        except KeyError as e:
            print(f"Error: Patient {p['patient_id']} is missing a required feature: {e}")
            continue

        # Predict
        pred = enc.inverse_transform(model.predict(df))[0]
        conf = round(max(model.predict_proba(df)[0]) * 100, 2)
        
        # Print result
        print(f"--- Patient: {p['patient_id']} ---")
        print(tabulate(df.T.reset_index().values, headers=["Feature", "Value"], tablefmt="simple"))
        print(f"Prediction: {pred} | Confidence: {conf}%\n")
        
        results.append([p['patient_id'], pred, conf])

    print(tabulate(results, headers=["ID", "Class", "Conf %"], tablefmt="fancy_grid"))

    export_html_report(results)

if __name__ == "__main__":
    main()