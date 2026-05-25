import numpy as np
import pandas as pd
import random

class_dist = {
    "non_urgent": 0.55,
    "urgent": 0.30,
    "emergency": 0.15
}


# PATIENT PHENOTYPES


phenotype_config = {

    "non_urgent": {

        "healthy": 0.60,
        "anxiety": 0.20,
        "athlete": 0.10,
        "chronic_stable": 0.10
    },

    "urgent": {

        "moderate_infection": 0.45,
        "respiratory_distress": 0.30,
        "dehydration": 0.15,
        "borderline_emergency": 0.10
    },

    "emergency": {

        "septic_shock": 0.35,
        "silent_hypoxia": 0.25,
        "cardiovascular_collapse": 0.20,
        "masked_elderly": 0.20
    }
}

# FEATURE CONFIG
# (mean, std, min_clip, max_clip)


feature_config = {

    "healthy": {

        "age": (40, 15, 18, 90),
        "hr": (72, 12, 45, 115),
        "rr": (14, 3, 8, 24),
        "systolic_bp": (124, 12, 90, 160),
        "spo2": (98, 2, 92, 100),
        "temp": (36.8, 0.5, 35.0, 38.5),
        "wbc_count": (7.0, 2.5, 3.0, 14.0)
    },

    "anxiety": {

        "age": (32, 12, 18, 75),
        "hr": (112, 18, 70, 155),
        "rr": (26, 6, 14, 42),
        "systolic_bp": (128, 15, 90, 180),
        "spo2": (98, 1, 94, 100),
        "temp": (36.9, 0.3, 35.5, 38.0),
        "wbc_count": (7.5, 2.0, 3.0, 13.0)
    },

    "athlete": {

        "age": (28, 8, 18, 50),
        "hr": (52, 8, 35, 80),
        "rr": (12, 2, 8, 18),
        "systolic_bp": (118, 10, 90, 145),
        "spo2": (99, 1, 95, 100),
        "temp": (36.7, 0.3, 35.5, 37.8),
        "wbc_count": (6.5, 1.8, 3.0, 11.0)
    },

    "chronic_stable": {

        "age": (65, 12, 40, 90),
        "hr": (88, 15, 55, 130),
        "rr": (20, 4, 10, 32),
        "systolic_bp": (132, 18, 85, 180),
        "spo2": (90, 4, 82, 97),
        "temp": (36.9, 0.4, 35.0, 38.0),
        "wbc_count": (8.0, 2.5, 3.0, 16.0)
    },

    "moderate_infection": {

        "age": (50, 18, 18, 90),
        "hr": (98, 18, 60, 150),
        "rr": (22, 5, 12, 36),
        "systolic_bp": (108, 16, 75, 160),
        "spo2": (93, 4, 80, 99),
        "temp": (38.2, 0.8, 36.0, 40.5),
        "wbc_count": (12.5, 4.0, 4.0, 24.0)
    },

    "respiratory_distress": {

        "age": (58, 16, 18, 90),
        "hr": (108, 20, 65, 165),
        "rr": (30, 7, 18, 50),
        "systolic_bp": (102, 15, 70, 150),
        "spo2": (88, 6, 72, 97),
        "temp": (37.8, 0.7, 35.8, 40.0),
        "wbc_count": (13.5, 4.5, 4.0, 28.0)
    },

    "dehydration": {

        "age": (45, 18, 18, 90),
        "hr": (102, 15, 65, 145),
        "rr": (20, 4, 10, 32),
        "systolic_bp": (92, 14, 65, 135),
        "spo2": (96, 2, 88, 100),
        "temp": (37.3, 0.5, 35.5, 39.0),
        "wbc_count": (9.5, 3.0, 3.0, 18.0)
    },

    "borderline_emergency": {

        "age": (60, 18, 18, 90),
        "hr": (118, 18, 75, 175),
        "rr": (28, 6, 14, 45),
        "systolic_bp": (92, 16, 60, 145),
        "spo2": (86, 5, 70, 97),
        "temp": (38.8, 0.8, 36.0, 41.0),
        "wbc_count": (15.0, 5.0, 5.0, 32.0)
    },

    "septic_shock": {

        "age": (64, 14, 18, 90),
        "hr": (132, 22, 80, 190),
        "rr": (34, 8, 18, 55),
        "systolic_bp": (78, 16, 50, 120),
        "spo2": (84, 6, 60, 95),
        "temp": (39.5, 1.0, 36.0, 42.0),
        "wbc_count": (18.0, 6.0, 6.0, 40.0)
    },

    "silent_hypoxia": {

        "age": (58, 15, 18, 90),
        "hr": (96, 16, 60, 150),
        "rr": (22, 5, 10, 40),
        "systolic_bp": (108, 14, 70, 150),
        "spo2": (78, 7, 55, 92),
        "temp": (37.8, 0.6, 35.5, 40.0),
        "wbc_count": (14.0, 4.5, 5.0, 30.0)
    },

    "cardiovascular_collapse": {

        "age": (70, 12, 35, 90),
        "hr": (126, 20, 70, 185),
        "rr": (30, 7, 12, 50),
        "systolic_bp": (68, 12, 45, 110),
        "spo2": (87, 5, 65, 96),
        "temp": (37.0, 0.5, 35.0, 39.0),
        "wbc_count": (11.5, 4.0, 4.0, 28.0)
    },

    "masked_elderly": {

        "age": (78, 8, 60, 95),
        "hr": (96, 14, 60, 140),
        "rr": (24, 5, 12, 38),
        "systolic_bp": (92, 14, 60, 135),
        "spo2": (89, 5, 72, 97),
        "temp": (37.2, 0.5, 35.0, 39.0),
        "wbc_count": (15.0, 4.0, 5.0, 32.0)
    }
}


# TEMPORAL CONFIG


temporal_config = {

    "delta_hr": {
        "non_urgent": (0, 5, -25, 25),
        "urgent": (5, 8, -20, 35),
        "emergency": (14, 12, -15, 60)
    },

    "delta_rr": {
        "non_urgent": (0, 3, -10, 10),
        "urgent": (3, 5, -10, 18),
        "emergency": (8, 8, -8, 30)
    },

    "delta_spo2": {
        "non_urgent": (0, 2, -8, 8),
        "urgent": (-2, 4, -15, 8),
        "emergency": (-6, 6, -30, 10)
    },

    "delta_systolic_bp": {
        "non_urgent": (0, 6, -25, 25),
        "urgent": (-4, 7, -35, 20),
        "emergency": (-12, 12, -70, 20)
    },

    "delta_temp": {
        "non_urgent": (0, 0.4, -2, 2),
        "urgent": (0.4, 0.5, -1, 3),
        "emergency": (1.0, 0.8, -1, 5)
    }
}


def sample_class():

    labels = list(class_dist.keys())
    probs = list(class_dist.values())

    return np.random.choice(labels, p=probs)



def sample_phenotype(class_label):

    phenotype_dict = phenotype_config[class_label]

    labels = list(phenotype_dict.keys())
    probs = list(phenotype_dict.values())

    return np.random.choice(labels, p=probs)



def generate_static_feature(params, feature_name=None):

    mean, std, min_clip, max_clip = params

    value = np.random.normal(mean, std)

    value = np.clip(value, min_clip, max_clip)

    noise = np.random.normal(0, std * 0.15)

    value += noise

    value = np.clip(value, min_clip, max_clip)

    if feature_name == "age":
        return int(abs(round(value)))

    return round(value, 2)



def generate_delta_feature(params, current_value=None, feature_name=None):

    mean, std, min_clip, max_clip = params


    coupling_strength = np.random.uniform(0.3, 1.2)

    if feature_name == "delta_hr":

        if current_value > 120:
            mean += (5 * coupling_strength)

        elif current_value < 75:
            mean -= (2 * coupling_strength)

    elif feature_name == "delta_spo2":

        if current_value < 88:
            mean -= (4 * coupling_strength)

    elif feature_name == "delta_rr":

        if current_value > 28:
            mean += (4 * coupling_strength)

    elif feature_name == "delta_systolic_bp":

        if current_value < 85:
            mean -= (6 * coupling_strength)


    delta = np.random.normal(mean, std)

    if np.random.rand() < 0.08:
        delta *= -1

    delta = np.clip(delta, min_clip, max_clip)

    return round(delta, 2)



def generate_comorbidity(class_label):

    if class_label == "non_urgent":
        return np.random.binomial(1, 0.18)

    elif class_label == "urgent":
        return np.random.binomial(1, 0.45)

    else:
        return np.random.binomial(1, 0.75)



def generate_crp(class_label):

    if class_label == "non_urgent":
        scale = 12

    elif class_label == "urgent":
        scale = 42

    else:
        scale = 90

    return round(np.random.exponential(scale), 2)



def generate_patient():

    patient = {}


    class_label = sample_class()

    patient["class"] = class_label



    phenotype = sample_phenotype(class_label)

    patient["phenotype"] = phenotype



    for feature in feature_config[phenotype]:

        params = feature_config[phenotype][feature]

        value = generate_static_feature(params, feature)

        patient[feature] = value


    for del_feature in temporal_config:

        params = temporal_config[del_feature][class_label]

        base_feature = del_feature.replace("delta_", "")

        current_value = patient[base_feature]

        delta_value = generate_delta_feature(
            params=params,
            current_value=current_value,
            feature_name=del_feature
        )

        patient[del_feature] = delta_value


    patient["comorbidity"] = generate_comorbidity(class_label)

    patient["crp"] = generate_crp(class_label)


    noise_probability = np.random.rand()

    if class_label == "urgent" and noise_probability < 0.04:
        patient["class"] = "emergency"

    elif class_label == "emergency" and noise_probability < 0.04:
        patient["class"] = "urgent"

    return patient



NUM_SAMPLES = 1000000

patients = []

for _ in range(NUM_SAMPLES):

    patient = generate_patient()

    patients.append(patient)

df = pd.DataFrame(patients)
df.to_csv("data/new_data3/triage_dataset_1m.csv", index=False)