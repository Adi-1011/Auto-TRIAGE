# Auto-TRIAGE 🏥

Auto-TRIAGE is a comparative machine learning project focused on synthetic symptom-based triage classification using clinically constrained probabilistic data generation.

The project was developed as part of a Final Year Research Project (FYRP) at  
**ITER, Siksha 'O' Anusandhan (Deemed to be) University**.

---

## Overview

Emergency triage systems often depend on fast and reliable prioritization of patients based on physiological severity. Access to real hospital triage datasets, however, is heavily restricted due to privacy and ethical concerns.

This project explores whether a carefully structured synthetic dataset can still support meaningful experimentation for triage classification.

Instead of purely random data generation, the framework introduces:
- probabilistic clinical simulation,
- temporal deterioration modeling,
- controlled physiological overlap,
- phenotype diversity,
- and progression-aware feature engineering.

The primary goal was to evaluate how different supervised machine learning models behave under increasingly realistic triage conditions while maintaining clinically plausible distributions.

---

## Project Pipeline

```text
Synthetic Clinical Data Generation
            ↓
Physiological Feature Engineering
            ↓
Temporal Deterioration Modeling
            ↓
Preprocessing + Stratified 5-Fold CV
            ↓
┌────────────┬────────────┬────────────┬────────────┐
│ Logistic   │ Random     │ MLP        │ XGBoost    │
│ Regression │ Forest     │            │            │
└────────────┴────────────┴────────────┴────────────┘
            ↓
Comparative Evaluation
(Accuracy, F1, Emergency Recall, ROC-AUC)
```

---

## Triage Classes

| Class | Description |
|---|---|
| Non-Urgent | Stable physiological condition |
| Urgent | Moderate instability requiring medical attention |
| Emergency | Critical condition requiring immediate intervention |

---

## Dataset Design

The dataset used in this project is fully synthetic and generated using clinically constrained probabilistic distributions instead of real patient records.

The final dataset incorporates:
- static physiological features,
- temporal delta features,
- controlled class overlap,
- ambiguity injection,
- and phenotype-based variability.

The framework was gradually refined across multiple dataset iterations to reduce unrealistic class separability and improve generalization behavior.

### Static Features
- Age
- Heart Rate
- Respiratory Rate
- Systolic Blood Pressure
- SpO₂
- Body Temperature
- WBC Count
- CRP Level
- Comorbidity Indicator

### Temporal Features
- ΔHR
- ΔRR
- ΔSpO₂
- ΔTemp
- ΔSBP

---

## Machine Learning Models

The following supervised learning models were evaluated:

- Logistic Regression
- Random Forest
- Multilayer Perceptron (MLP)
- XGBoost

All models were trained and evaluated using stratified 5-fold cross-validation under identical experimental settings.

---

## Experimental Results

| Model | Accuracy | Emergency Recall |
|---|---|---|
| Logistic Regression | 92.18% | 76.91% |
| Random Forest | 95.41% | 85.70% |
| MLP | 95.45% | 86.10% |
| XGBoost | **95.80%** | **85.82%** |

XGBoost achieved the best overall balance between predictive performance and computational efficiency.

---

## Repository Structure

```text
Auto-TRIAGE/
│
├── backend/
│   ├── api_endpoint/
│   │   ├── main.py
│   │   ├── index.html
│   │   └── style.css
│   │
│   └── llm/
│       ├── feature_extract.py
│       └── condition_1.jpg
│
├── data/
│   ├── raw/
│   ├── processed2/
│   └── Images/
│
├── evaluation/
│   ├── 1m/
│   │   ├── Dataset Workflow.png
│   │   ├── Proposed System Architecture.png
│   │   └── Temporal Deterioration Modeling.png
│   │
│   └── cross_validation.py
│
├── generator/
│   ├── synthetic_data_gen.py
│   ├── data_gen2.py
│   └── data_gen3.py
│
├── models/
│   ├── logistic_reg.py
│   ├── random_forest.py
│   ├── mlpclassifier.py
│   ├── xgb.py
│   │
│   ├── saved/
│   ├── saved2/
│   └── saved3/
│
├── results/
├── results3/
├── new_results/
│
├── visualizer/
│   └── represent_data.py
│
├── working_models/
│   ├── runxgb.py
│   └── run_xgb_pred.bat
│
├── manuscript/
│
├── api_test.py
├── requirements.txt
└── README.md
```

---

## Local Setup

### Clone Repository

```bash
git clone https://github.com/Adi-1011/Auto-TRIAGE.git
cd Auto-TRIAGE
```

### Create Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Generate Synthetic Dataset

```bash
python generator/synthetic_data_gen.py
```

### Run Visualizations

```bash
python visualizer/represent_data.py
```

### Train Models

Example:

```bash
python models/xgb.py
```

Other model scripts can be executed similarly.

---

## Testing Saved Models

The repository also includes a lightweight testing setup for running trained XGBoost models with demo input data.

### Python Inference Script

```bash
python working_models/runxgb.py
```

### Windows Batch Script

```bash
run_xgb_pred.bat
```

This loads the saved model and performs prediction on sample patient input data.

---

## Research Notes

This repository is intended primarily for:
- experimentation,
- comparative ML evaluation,
- and research-oriented exploration.

It is not intended for real clinical deployment or medical decision-making.

---

## Branch Information

The most recent experimental pipeline and manuscript updates are maintained in:

```text
feature/latest
```

---

## Team

**Group 02-12 — B.Tech CSE**

- Aditya Kumar
- Aditya Kumar
- Aman Kumar
- Arav Prasad

### Supervisor
Dr. Ajay Shankar Tiwari

Department of Computer Science and Engineering  
ITER, Siksha 'O' Anusandhan (Deemed to be) University

---

## License

This project is intended for academic and research purposes only.