# Auto-TRIAGE 🏥

Auto-TRIAGE is a comparative machine learning project focused on synthetic symptom-based triage classification using clinically constrained probabilistic data generation.

The project was developed as part of a Final Year Research Project (FYRP) at  
**ITER, Siksha 'O' Anusandhan (Deemed to be) University**.

---

## Overview

Access to real clinical triage datasets is heavily restricted due to privacy and ethical concerns. This project explores whether a carefully designed synthetic dataset can still support meaningful experimentation for emergency triage classification.

Instead of relying on purely random data generation, the framework attempts to simulate clinically plausible patient behavior using:
- class-conditional probabilistic modeling,
- physiological constraints,
- temporal deterioration patterns,
- controlled overlap between severity classes,
- and phenotype-based variability.

The goal was not to create a production medical system, but to study how different supervised machine learning models behave under progressively more realistic triage conditions.

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
| Urgent | Moderate instability requiring clinical attention |
| Emergency | Critical condition requiring immediate intervention |

---

## Dataset Design

The dataset used in this project is fully synthetic and generated using probabilistic distributions instead of real patient records.

The final dataset includes:
- static physiological features,
- temporal progression features,
- controlled clinical ambiguity,
- overlap between neighboring severity classes,
- and phenotype diversity to reduce unrealistic separability.

### Physiological Features

#### Static Features
- Age
- Heart Rate
- Respiratory Rate
- Systolic Blood Pressure
- SpO₂
- Body Temperature
- WBC Count
- CRP Level
- Comorbidity Indicator

#### Temporal Features
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

XGBoost achieved the best overall balance between classification performance and computational efficiency.

---

## Repository Structure

```text
Auto-TRIAGE/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── generator/
│   └── synthetic_data_gen.py
│
├── models/
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── mlp/
│   └── xgboost/
│
├── visualizer/
│   └── visualize.py
│
├── manuscript/
│
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

### Generate Dataset

```bash
python generator/synthetic_data_gen.py
```

### Run Visualizations

```bash
python visualizer/visualize.py
```

### Train Models

Run the corresponding training scripts from the model directories.

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