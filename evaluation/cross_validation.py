import time
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb


DATASET = "data/new_data3/triage_dataset_1m.csv"

df = pd.read_csv(DATASET)

X = df.drop(["class", "phenotype"], axis=1)
y_text = df["class"]

le = LabelEncoder()
y = le.fit_transform(y_text)

print("Dataset:", DATASET)
print("Dataset shape:", df.shape)
print("Class order:", le.classes_)
print()


models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=500,
            random_state=42,
            verbose=1,
            solver="saga"
        ))
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=200,
            tol=0.00001,
            n_iter_no_change=50,
            learning_rate="adaptive",
            random_state=42,
            verbose=True
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=42,
        max_depth=None,
        n_jobs=6,
        verbose=1
    ),

    "XGBoost": xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        max_depth=6,
        n_estimators=100,
        n_jobs=6,
        verbosity=1,
        eval_metric="mlogloss",
        random_state=42,
        learning_rate=0.3
    )
}

# 5-FOLD CROSS VALIDATION

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_results = []

for model_name, model in models.items():

    print("=" * 60)
    print(f"Running 5-Fold Cross Validation for: {model_name}")
    print("=" * 60)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        current_model = clone(model)

        start_time = time.time()
        current_model.fit(X_train, y_train)
        end_time = time.time()

        y_pred = current_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        class_recalls = recall_score(y_test, y_pred, average=None, zero_division=0)

        result = {
            "model": model_name,
            "fold": fold,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "emergency_recall": class_recalls[0],
            "non_urgent_recall": class_recalls[1],
            "urgent_recall": class_recalls[2],
            "training_time_sec": end_time - start_time
        }

        fold_results.append(result)
        all_results.append(result)

        print(f"\nFold {fold} completed")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Emergency Recall: {class_recalls[0]:.4f}")
        print(f"Urgent Recall: {class_recalls[2]:.4f}")
        print(f"Training Time: {end_time - start_time:.2f} sec")

# FINAL SUMMARY

results_df = pd.DataFrame(all_results)

summary_df = results_df.groupby("model").agg({
    "accuracy": ["mean", "std"],
    "macro_precision": ["mean", "std"],
    "macro_recall": ["mean", "std"],
    "macro_f1": ["mean", "std"],
    "weighted_f1": ["mean", "std"],
    "emergency_recall": ["mean", "std"],
    "non_urgent_recall": ["mean", "std"],
    "urgent_recall": ["mean", "std"],
    "training_time_sec": ["mean", "std"]
})

print("\n\n================ FINAL 5-FOLD CV SUMMARY ================")
print(summary_df)

results_df.to_csv("evaluation/1m/cross_validation_fold_results.csv", index=False)
summary_df.to_csv("evaluation/1m/cross_validation_summary.csv")

print("\nResults saved to:")
print("evaluation/1m/cross_validation_fold_results.csv")
print("evaluation/1m/cross_validation_summary.csv")
