import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn import metrics
import time

dataset = "data/raw/triage_dataset_1m.csv"

df = pd.read_csv(dataset)

X = df.drop('triage_class', axis=1)
Y = df['triage_class']

le = LabelEncoder()
Y = le.fit_transform(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#training the model
start_boosting = time.time()
boost = xgb.XGBClassifier(objective='multi:softmax', num_class = 3,max_depth = 6, n_estimators=100,n_jobs=6, verbosity = 1, eval_metric = 'mlogloss',random_state = 42,learning_rate = 0.3)
boost.fit(
    x_train, y_train,
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
print(f"Time taken: {(end_boosting - start_boosting)} seconds")
print(f"Time taken: {(end_boosting - start_boosting) / 60:.2f} minutes")