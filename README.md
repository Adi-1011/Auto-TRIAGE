# Auto-TRIAGE 🏥

> **A Controlled Experimental Study on Machine Learning-Based Triage Classification Using Probabilistic Synthetic Data**

A research-oriented Final Year Project (FYP) conducted at **Sikhsa 'O' Anusandhan University, ITER — Department of Computer Science and Engineering**.

This project investigates whether a controlled comparative study of ML models trained on probabilistically generated synthetic clinical data can provide insights into model suitability for safety-prioritized triage classification.

---

## 🔬 Research Scope

This is **not** a production application. It is a controlled ML experiment with the following pipeline:

```
Probabilistic Synthetic Dataset (9 clinical features)
                    ↓
        Red-Flag Rule Layer (deterministic safety pre-filter)
                    ↓
         Train / Test Split + k-Fold Cross Validation
                    ↓
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │    Logistic  │    Random    │   XGBoost    │     MLP      │
  │  Regression  │    Forest    │  (Boosting)  │   (Neural)   │
  └──────────────┴──────────────┴──────────────┴──────────────┘
                    ↓
     Comparative Performance Evaluation
     (Accuracy, Precision, Recall, F1, ROC-AUC)
     Primary focus: Recall for Emergency class
```

### Triage Classes
| Class | Clinical Meaning |
|---|---|
| `Non-Urgent` | Home care / Pharmacy / Routine consultation |
| `Urgent` | Clinical visit / Advanced care |
| `Emergency` | ER visit / Immediate hospitalization |

---

## 📁 Repository Structure

```
Auto-TRIAGE/
├── data/
│   ├── raw/            ← Generated synthetic datasets (CSV, gitignored)
│   └── processed/      ← Cleaned data and visualization outputs
├── generator/
│   └── synthetic_data_gen.py   ← Probabilistic dataset generator
├── visualizer/
│   └── visualize.py            ← Feature distribution and EDA plots
├── .gitignore
└── requirememts.txt
```

> **Note:** `data/raw/*.csv` and `data/processed/*.png` are gitignored. Run the generator locally to produce the dataset.

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Adi-1011/Auto-TRIAGE.git
cd Auto-TRIAGE
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirememts.txt
```

### 4. Generate the Synthetic Dataset

```bash
python generator/synthetic_data_gen.py
```

This will generate `triage_dataset.csv` inside `data/raw/`.

### 5. Run Visualizations

```bash
python visualizer/visualize.py
```

Plots will be saved to `data/processed/`.

---

## 🤝 Contributing

### Branch Workflow

All contributions must be made through feature branches. **Do not push directly to `main`.**

```bash
# Step 1 - Pull latest main
git checkout main
git pull origin main

# Step 2 - Create your feature branch
git checkout -b feature/your-feature-name

# Step 3 - Make your changes, then stage and commit
git add .
git commit -m "your descriptive commit message"

# Step 4 - Push your branch
git push origin feature/your-feature-name

# Step 5 - Open a Pull Request on GitHub for review
```

### Branch Naming Convention
| Type | Example |
|---|---|
| New feature | `feature/logistic-regression-training` |
| Bug fix | `fix/label-encoding-issue` |
| Evaluation | `eval/cross-validation-results` |
| Documentation | `docs/update-readme` |

---

## 🧪 Dataset Design

The synthetic dataset is generated using clinically grounded probabilistic distributions:

| Feature | Distribution | Range |
|---|---|---|
| Age | Uniform | 18 – 90 |
| Heart Rate | Normal (μ=80, σ=20) | 40 – 160 bpm |
| Respiratory Rate | Normal (μ=16, σ=5) | 8 – 40 bpm |
| Systolic BP | Normal (μ=115, σ=20) | 70 – 200 mmHg |
| SpO₂ | Normal (μ=97, σ=2) | 75 – 100 % |
| Body Temperature | Normal (μ=37, σ=1) | 34 – 41 °C |
| WBC Count | Normal (μ=7.5, σ=3) | 2 – 25 ×10⁹/L |
| CRP Level | Exponential (λ=1/20) | 0 – 200 mg/L |
| Comorbidity Flag | Bernoulli (p=0.3) | 0 / 1 |

Labels are assigned via a **deterministic red-flag safety layer** followed by **probabilistic severity scoring** for non-critical cases.

---

## 👥 Team

**Group 02-12 — B.Tech CSE, 8th Semester**
Sikhsa 'O' Anusandhan University (ITER)

---

## 📄 License

This project is for academic research purposes only.
