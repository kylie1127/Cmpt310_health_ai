import pickle
import pandas as pd
import numpy as np

symptoms_to_test = ["headache", "depression", "sore throat"]

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

def run_prediction(symptoms_list):
    user_input = [s.strip().lower() for s in symptoms_list]

    input_vector = pd.DataFrame(0, index=[0], columns=feature_names)
    
    matched_count = 0
    for s in user_input:
        if s in input_vector.columns:
            input_vector.at[0, s] = 1
            matched_count += 1

    if "symptom_count" in input_vector.columns:
        input_vector.at[0, "symptom_count"] = matched_count

    probabilities = model.predict_proba(input_vector)[0]
    results = pd.DataFrame({'Disease': model.classes_, 'Prob': probabilities})
    results = results.sort_values('Prob', ascending=False).head(5)

    print(f"\nRESULTS FOR: {', '.join(symptoms_list)}")
    print("-" * 50)
    for _, row in results.iterrows():
        print(f"{row['Disease']:<35} | {row['Prob']*100:>8.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    run_prediction(symptoms_to_test)