import numpy as np 
import pandas as pd 
import time

class_dist = {
    "non_urgent": 0.55,
    "urgent": 0.30,
    "emergency": 0.15
}

feature_config = {
    "age": {
        "non_urgent": (50, 18, 18, 90),
        "urgent": (50, 18, 18, 90),
        "emergency":(50, 18, 18, 90)
    },
    "hr": {
        "non_urgent": (75,10,45,105),
        "urgent": (98, 14, 70, 140),
        "emergency":(128, 18, 90, 190)
    },
    "rr": {
        "non_urgent": (14, 3, 8, 22),
        "urgent": (22, 4, 14, 32),
        "emergency":(31, 6, 18, 45)
    },
    "systolic_bp": {
        "non_urgent": (122, 10, 90, 150),
        "urgent": (103, 13, 75, 140),
        "emergency":(82, 14, 55, 120)
    },
    "spo2": {
        "non_urgent": (98, 1, 93, 100),
        "urgent": (93, 3, 82, 99),
        "emergency":(85, 5, 65, 96)
    },
    "temp": {
        "non_urgent": (36.9, 0.4, 35.0, 38.0),
        "urgent": (38.1, 0.7, 36.0, 40.0),
        "emergency":(39.4, 0.9, 36.5, 42.0)
    },
    "wbc_count": {
        "non_urgent": (7.0, 1.8, 3.0, 12.0),
        "urgent": (11.5, 3.0, 5.0, 20.0),
        "emergency":(16.5, 4.5, 7.0, 35.0)
    }
}

temporal_config = {

    "delta_hr": {
        # (mean_delta, std_delta, min_clip, max_clip)
        "non_urgent": (0, 3, -15, 15),
        "urgent": (8, 5, -10, 25),
        "emergency": (22, 8, -5, 55)
    },

    "delta_rr": {
        "non_urgent": (0, 2, -8, 8),
        "urgent": (4, 3, -5, 15),
        "emergency": (10, 5, 0, 25)
    },

    "delta_spo2": {
        "non_urgent": (0, 1, -5, 5),
        "urgent": (-3, 2, -12, 5),
        "emergency": (-8, 4, -25, 3)
    },

    "delta_systolic_bp": {
        "non_urgent": (0, 4, -20, 20),
        "urgent": (-6, 5, -30, 15),
        "emergency": (-18, 8, -60, 10)
    },

    "delta_temp": {
        "non_urgent": (0, 0.2, -1, 1),
        "urgent": (0.5, 0.3, -0.5, 2),
        "emergency": (1.2, 0.5, -0.2, 4)
    }
}

def sample_class():
    labels = list(class_dist.keys())
    probs = list(class_dist.values())

    return np.random.choice(labels,p=probs)

def generate_static_feature(params, feature_name=None):

    mean, std, min_clip, max_clip = params
    value = np.random.normal(mean,std)
    value = np.clip(value, min_clip, max_clip)
    if feature_name == "age":
        return int(abs(round(value)))
    return round(value,2)

def generate_delta_feature(params, current_value= None, feature_name = None):
    mean, std, min_clip, max_clip = params
    if feature_name == 'delta_hr':
        if current_value > 120:
            mean+=6
        elif current_value < 80:
            mean-=3

    elif feature_name == "delta_spo2":
        if current_value < 88:
            mean -= 4

    elif feature_name == "delta_rr":
        if current_value > 28:
            mean += 4

    elif feature_name == "delta_systolic_bp":
        if current_value < 85:
            mean -= 6

    delta = np.random.normal(mean, std)
    delta = np.clip(delta, min_clip, max_clip)
    return round(delta,2)

def generate_patient():
    patient = {}

    class_label = sample_class()
    patient['class'] = class_label

    for feature in feature_config:
        params = feature_config[feature][class_label]
        value = generate_static_feature(params, feature)
        patient[feature] = value

    for del_feature in temporal_config:
        params = temporal_config[del_feature][class_label]
        # Map delta feature to current feature
        base_feature = del_feature.replace("delta_", "")

        current_value = patient[base_feature]

        delta_value = generate_delta_feature(
            params=params,
            current_value=current_value,
            feature_name=del_feature
        )

        patient[del_feature] = delta_value

    # generating comorbidity
    if class_label == "non_urgent":
        patient["comorbidity"] = np.random.binomial(1, 0.15)
    elif class_label == "urgent":
        patient["comorbidity"] = np.random.binomial(1, 0.40)
    else:
        patient["comorbidity"] = np.random.binomial(1, 0.70)

    # generating CRP
    if class_label == "non_urgent":
        scale = 8
    elif class_label == "urgent":
        scale = 35
    else:
        scale = 85
    patient["crp"] = round(np.random.exponential(scale), 2)

    return patient

dataset = []
samples = 100000
for _ in range(samples):
    patient = generate_patient()
    dataset.append(patient)

df = pd.DataFrame(dataset)
df.to_csv("data/new_data/triage_dataset_100k.csv", index=False)