import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/small_scale_triage_dataset.csv")

features = ["age", "heart_rate", "respiratory_rate", "systolic_bp", 
            "spo2", "temprature", "wbc", "crp", "comorbidty"]

# Step 1 - Basic info
print(df.shape)
print(df.head())
print(df["triage_class"].value_counts())

# Step 2 - Feature distributions
for feature in features:
    plt.figure()
    plt.hist(df[feature], bins=30, color="steelblue", edgecolor="black")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.savefig(f"data/processed/{feature}_distribution.png")
    plt.show()
    plt.close()

# Step 3 - Triage class distribution
plt.figure()
df["triage_class"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Triage Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("data/processed/triage_class_distribution.png")
plt.show()
plt.close()

# Step 4 - Box plots grouped by triage class
for feature in features:
    plt.figure()
    df.boxplot(column=feature, by="triage_class")
    plt.title(f"{feature} by Triage Class")
    plt.suptitle("")
    plt.xlabel("Triage Class")
    plt.ylabel(feature)
    plt.savefig(f"data/processed/{feature}_boxplot.png")
    plt.show()
    plt.close()

print("All plots saved to data/processed/")