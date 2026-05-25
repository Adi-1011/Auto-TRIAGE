import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/new_data3/triage_dataset_1m.csv")

# Features to visualize
features = [
    "age",
    "hr",
    "rr",
    "systolic_bp",
    "spo2",
    "temp",
    "wbc_count",
    "crp",
    "delta_hr",
    "delta_rr",
    "delta_spo2",
    "delta_systolic_bp",
    "delta_temp"
]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

scaled_df = pd.DataFrame(scaled_data, columns=features)
scaled_df["class"] = df["class"]


plt.figure(figsize=(16, 8))


melted = scaled_df.melt(
    id_vars="class",
    var_name="Feature",
    value_name="Scaled Value"
)

# Boxplot grouped by feature and class
import seaborn as sns

sns.boxplot(
    data=melted,
    x="Feature",
    y="Scaled Value",
    hue="class"
)

plt.xticks(rotation=45)
plt.title("Feature Distribution Across Triage Classes")
plt.tight_layout()

plt.savefig("data/processed2/combined_feature_boxplot.png", dpi=300)

plt.show()