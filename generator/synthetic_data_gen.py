import numpy as np 
import pandas as pd 
import time

start = time.time()

rows = []

for i in range(200000):

    age = np.random.uniform(low=0, high=100)
    heart_rate = np.random.normal(loc=80, scale=20)
    respiratory_rate = np.random.normal(loc=16, scale=5)
    blood_pressure = np.random.normal(loc=115, scale=20)
    spo2 = np.random.normal(loc=97, scale=2)
    tempreature = np.random.normal(loc=37, scale=1)
    wbc = np.random.normal(loc=7.5, scale=3)
    crp = np.random.exponential(scale=20)
    comorbidty = np.random.choice([0,1], p=[0.7,0.3])

    clipped_age = age
    clipped_hr = np.clip(heart_rate, 40, 160)
    clipped_rr = np.clip(respiratory_rate, 8, 40)
    clipped_bp = np.clip(blood_pressure, 70, 200)
    clipped_spo2 = np.clip(spo2, 75, 100)
    clipped_temp = np.clip(tempreature, 34, 41)
    clipped_wbc = np.clip(wbc, 2, 25)
    clipped_crp = crp
    clipped_comorbidty = comorbidty

    # writing red flag determnistic logic 

    if clipped_spo2 < 90 or clipped_bp < 90 or clipped_hr > 150 or clipped_rr > 35 or clipped_temp > 40:
        label = "Emergency"

    else:

        severity_score = 0

        if clipped_hr > 110: severity_score+=2 
        if clipped_rr > 24: severity_score+=2 
        if clipped_spo2 < 94: severity_score+=2 
        if clipped_temp > 38: severity_score+=1 
        if clipped_wbc > 11: severity_score+=1 
        if clipped_crp > 40: severity_score+=2 
        if clipped_age > 70: severity_score+=1 
        if clipped_comorbidty == 1: severity_score+=1 

        if severity_score <=2:
            prob = [0.80, 0.18, 0.02]
        elif severity_score <=5:
            prob = [0.35, 0.55, 0.10]
        elif severity_score <=8:
            prob = [0.10, 0.65, 0.25]
        else:
            prob = [0.02, 0.45, 0.53]

        label = np.random.choice(["Non-urgent", "Urgent", "Emergency"], p = prob)

    rows.append({
        "age": clipped_age,
        "heart_rate": clipped_hr,
        "respiratory_rate": clipped_rr,
        "systolic_bp": clipped_bp,
        "spo2": clipped_spo2,
        "temprature": clipped_temp,
        "wbc": clipped_wbc,
        "crp": clipped_crp,
        "comorbidty": clipped_comorbidty,
        "triage_class": label
    })

df = pd.DataFrame(rows)
df.to_csv("data/raw/triage_dataset_200k.csv", index = False)

end = time.time()

print(f"Time taken: {end-start:.2f} seconds")


