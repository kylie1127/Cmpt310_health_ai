import pandas as pd

df = pd.read_csv("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")

print("Data shape:", df.shape)

X = df.drop("diseases", axis=1)
y = df["diseases"]

print("Feature shape:", X.shape)
print("Label shape:", y.shape)


X["symptom_count"] = X.sum(axis=1)

print("Added symptom_count feature")

X.to_csv("X_features.csv", index=False)
y.to_csv("y_labels.csv", index=False)

print("Saved X_features.csv and y_labels.csv")