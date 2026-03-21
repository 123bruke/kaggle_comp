# ============================================
# 📦 1. Import Libraries
# ============================================

import pandas as pd
import numpy as np

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


X_train = pd.read_csv("train.csv", header=None)
y_train = pd.read_csv("trainLabels.csv", header=None)


X_test = pd.read_csv("test.csv", header=None)


y_train = y_train.values.ravel()

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


print("\nFirst 5 rows of training data:")
print(X_train.head())

print("\nCheck missing values:")
print(X_train.isnull().sum().sum())


model = RandomForestClassifier(
    n_estimators=200,     # number of trees
    max_depth=None,       # allow full depth
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

model.fit(X_tr, y_tr)

y_pred_val = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred_val)
print("\nValidation Accuracy:", accuracy)

# Cross-validation (better evaluation)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation accuracy:", cv_scores.mean())

final_predictions = model.predict(X_test)

print("\nPrediction completed!")


submission = pd.DataFrame({
    "Id": range(1, len(final_predictions) + 1),
    "Solution": final_predictions
})

submission.to_csv("submission.csv", index=False)

print("\n✅ Submission file created: submission.csv")

importances = model.feature_importances_

plt.figure(figsize=(10,5))
plt.plot(importances)
plt.title("Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()