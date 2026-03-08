import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

data = pd.read_csv("data/cleaned_data.csv")

data = data.copy()

symptom_cols = data.columns.drop("diseases")
data['symptom_count'] = data[symptom_cols].sum(axis=1)

X = data.drop("diseases", axis=1)
y = data["diseases"]

feature_names = X.columns.tolist()
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)