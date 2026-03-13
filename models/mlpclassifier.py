from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import pandas as pd
import time
from tqdm import tqdm

start = time.time()
df = pd.read_csv("data/raw/triage_dataset_200k.csv")

X = df.drop('triage_class', axis=1)
y = df['triage_class']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, tol= 0.0000001, n_iter_no_change=50, learning_rate= 'adaptive', random_state=42, verbose=True)
mlp.fit(X_train, Y_train)

print(f"Stopped at epoch: {mlp.n_iter_}")
print(f"Final loss: {mlp.loss_:.6f}")

y_pred = mlp.predict(X_test)
print(metrics.classification_report(Y_test, y_pred, target_names=le.classes_))
end = time.time()
print(f"Time taken: {end-start:.2f} seconds")


