import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import time

def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()

dataset = "data/raw/triage_dataset_1m.csv"

df = pd.read_csv(dataset)
#printing data
print("dataset length:",len(df))
print("data shape:",df.shape)
print("Dataset head:\n",df.head())
# train test split
X = df.drop('triage_class', axis = 1)
Y = df['triage_class']
le = LabelEncoder()
Y = le.fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

start_gini = time.time()
# training using gini
# clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100,max_depth=None, min_samples_leaf=5)
clf_gini = RandomForestClassifier(n_estimators=100,criterion='gini', random_state=42,max_depth=None, n_jobs=6,verbose=1)
clf_gini.fit(x_train,y_train)
end_gini = time.time()
# training using Entropy
# clf_entropy = DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=None,min_samples_leaf=5)

start_entropy = time.time()
clf_entropy = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42,max_depth=None,n_jobs=6,verbose=1)
clf_entropy.fit(x_train,y_train)
end_entropy = time.time()

# predicting values gini
# y_pred_gini = clf_gini.predict(x_test)
# print("Predited Values:",y_pred_gini)

# predicting values entropy
y_pred_entropy = clf_entropy.predict(x_test)
print("Predited Values:",y_pred_entropy)


# calculatng accuracy
print("\nConfusion Matrix\n", confusion_matrix(y_test,y_pred_entropy))
print("\nAccuracy Score\n", accuracy_score(y_test,y_pred_entropy)*100)
print("\nClassification report\n", classification_report(y_test,y_pred_entropy))
print(f"Time taken: {(end_entropy - start_entropy)} seconds")
print(f"Time taken: {(end_entropy - start_entropy) / 60:.2f} minutes")

# plotting decision tree
# plot_decision_tree(
#     clf_gini,
#     X.columns,
#     list(map(str, clf_gini.classes_))
# )
#plotting first few trees from RF
plt.figure(figsize=(20, 10))
plot_tree(
    clf_entropy.estimators_[0],        # picks first tree from the forest
    filled=True,
    feature_names=X.columns,
    class_names=le.classes_,
    rounded=True,
    max_depth=3               # limit depth for readability
)
plt.title("Single Tree from Random Forest (Tree 0)")
plt.savefig("results/RF_Training_Results/sample_1m_entropy/rf_single_tree.png", dpi=150, bbox_inches='tight')
importances = clf_entropy.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances, color='steelblue', edgecolor='black')
plt.xlabel("Importance Score")
plt.title("Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig("results/RF_Training_Results/sample_1m/rf_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()