import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn import metrics
import time
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset = "data/new_data3/triage_dataset_1m.csv"

df = pd.read_csv(dataset)

X = df.drop(['class','phenotype'], axis=1)
print(X.columns.tolist())

Y = df['class']

le = LabelEncoder()
Y = le.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

#training the model
start_boosting = time.time()

boost = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class = 3,
    max_depth = 6,
    n_estimators=100,
    n_jobs=6,
    verbosity = 1,
    eval_metric = 'mlogloss',
    random_state = 42,
    learning_rate = 0.3
)

boost.fit(
    x_train,
    y_train,
    eval_set=[(x_test, y_test)],
    verbose=True
)

end_boosting = time.time()

#predictiing outcome
y_pred = boost.predict(x_test)
print("Prediction:",y_pred)

# calculatng accuracy
print("\nConfusion Matrix\n", confusion_matrix(y_test,y_pred))

print("\nAccuracy Score\n", accuracy_score(y_test,y_pred)*100)

print("\nClassification report\n", classification_report(y_test,y_pred))

# ROC AUC
macro_auc = roc_auc_score(
    label_binarize(y_test, classes=[0,1,2]),
    boost.predict_proba(x_test),
    multi_class='ovr',
    average='macro'
)

print(f"\nMacro Average ROC-AUC: {macro_auc:.4f}")

print(f"Time taken: {(end_boosting - start_boosting)} seconds")
print(f"Time taken: {(end_boosting - start_boosting) / 60:.2f} minutes")

# ROC Curve

classes = [0,1,2]

y_test_bin = label_binarize(y_test, classes=classes)

y_prob = boost.predict_proba(x_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

class_names = ["Emergency", "Non-Urgent", "Urgent"]

plt.figure(figsize=(8,6))

for i in range(len(classes)):

    fpr[i], tpr[i], _ = roc_curve(
        y_test_bin[:, i],
        y_prob[:, i]
    )

    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(
        fpr[i],
        tpr[i],
        label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})"
    )

plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")

plt.legend(loc="lower right")

plt.grid(True)

plt.savefig(
    "results3/images/xgboost_roc_curve.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# Feature Importance

xgb.plot_importance(boost, max_num_features=10)

plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "results3/images/xgboost_feature_importance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# Confusion Matrix Heatmap

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("XGBoost Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results3/images/xgboost_confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

import joblib

# joblib.dump(boost, "models/saved3/xgb/xgb_model.pkl")
# joblib.dump(le, "models/saved3/xgb/label_encoder.pkl")

print("XGB model saved succesfully")