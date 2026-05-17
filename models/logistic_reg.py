import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import pandas as pd

# change this to switch between datasets
DATASET = "data/new_data3/triage_dataset_1m.csv" 

df = pd.read_csv(DATASET)

X = df.drop(['class','phenotype'], axis=1)
y = df['class']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start = time.time()

lr = LogisticRegression(max_iter=500, random_state=42, verbose=1, solver="saga")
lr.fit(X_train, y_train)

end = time.time()

y_pred = lr.predict(X_test)

print(f"\nTotal iterations run: {lr.n_iter_}")
print(f"Converged: {lr.n_iter_ < 500}")
print(f"Time taken: {end - start:.2f} seconds")
print(f"Time taken: {(end - start) / 60:.2f} minutes")
print()
print(metrics.classification_report(y_test, y_pred, target_names=le.classes_))