import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── 1. Load data and remove classes with <2 samples
X = pd.read_csv("X_features.csv")
y = pd.read_csv("y_labels.csv").squeeze()

print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Classes: {y.unique()}")

class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 10].index
mask = y.isin(valid_classes)
X = X[mask]
y = y[mask]

# ── 2. Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ── 3. Train baseline decision tree
model = DecisionTreeClassifier(max_depth=100, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

print("\nBaseline model trained!")

# ── 4. Evaluate 
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy:  {acc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nFull Report:")
print(classification_report(y_test, y_pred))

# ── 5. Tune with Randomized Search 
params = {
    "max_depth": [20, 40, 60, 80, 100],
    "min_samples_split": [5, 10, 20]
}

grid = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    params,
    n_iter=6,        # only tries 6 random combos instead of all of them
    cv=3,            # 3 folds instead of 5
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1        # uses all your CPU cores
)

grid.fit(X_sample, y_sample)  
print(f"Best params: {grid.best_params_}")

# Retrain with best params
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

print(f"\nTuned Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Tuned F1 Score: {f1_score(y_test, y_pred_best, average='weighted'):.4f}")

# ── 6. Save model 
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nmodel.pkl saved — ready for Karan!")