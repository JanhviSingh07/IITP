"""
train_cost_model.py
--------------------
C-EGSR ka cost model train karta hai.
Heterogeneous signals se predict karta hai ki EGSR worthwhile hai ya nahi.

Run: python train_cost_model.py
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle, os

# Load data
with open("results/signal_data_combined.json") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
print(f"EGSR helped (positive): {sum(1 for d in data if d['egsr_helped'])}")
print(f"EGSR did NOT help (negative): {sum(1 for d in data if not d['egsr_helped'])}")
print()

# Feature matrix
FEATURES = [
    "cos_sim_top1",
    "cos_sim_top2", 
    "cos_sim_gap",
    "cos_sim_variance",
    "llm_confidence",
    "sql_complexity_r1",
    "question_length",
    "num_tables_in_db",
    "result_rows_r1",
]

X = np.array([[d[f] for f in FEATURES] for d in data])
y = np.array([1 if d["egsr_helped"] else 0 for d in data])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Feature Importance Analysis ===")
# Simple correlation with label
for i, feat in enumerate(FEATURES):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    print(f"  {feat}: correlation = {corr:.4f}")
print()

# Train models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=3, class_weight="balanced"),
}

# Leave-One-Out CV (best for small datasets)
loo = LeaveOneOut()

print("=== Model Comparison (Leave-One-Out CV) ===")
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    if name == "Logistic Regression":
        scores = cross_val_score(model, X_scaled, y, cv=loo, scoring="f1")
    else:
        scores = cross_val_score(model, X, y, cv=loo, scoring="f1")
    
    mean_f1 = scores.mean()
    print(f"  {name}: F1 = {mean_f1:.4f}")
    
    if mean_f1 > best_score:
        best_score = mean_f1
        best_name = name
        best_model = model

print(f"\nBest model: {best_name} (F1 = {best_score:.4f})")

# Train best model on full data
print("\n=== Training Best Model on Full Data ===")
if best_name == "Logistic Regression":
    best_model.fit(X_scaled, y)
else:
    best_model.fit(X, y)

# Feature importance (for Random Forest/Decision Tree)
if hasattr(best_model, "feature_importances_"):
    print("\nFeature Importances:")
    importances = best_model.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
elif hasattr(best_model, "coef_"):
    print("\nFeature Coefficients:")
    coeffs = best_model.coef_[0]
    for feat, coef in sorted(zip(FEATURES, coeffs), key=lambda x: -abs(x[1])):
        print(f"  {feat}: {coef:.4f}")

# Save model
os.makedirs("results", exist_ok=True)
model_data = {
    "model": best_model,
    "scaler": scaler if best_name == "Logistic Regression" else None,
    "features": FEATURES,
    "model_name": best_name,
    "best_f1": best_score,
    "needs_scaling": best_name == "Logistic Regression",
}
with open("results/cost_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to results/cost_model.pkl")

# Threshold analysis
print("\n=== Threshold Analysis ===")
if best_name == "Logistic Regression":
    probs = best_model.predict_proba(X_scaled)[:, 1]
else:
    probs = best_model.predict_proba(X)[:, 1]

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = (probs >= threshold).astype(int)
    tp = sum((preds == 1) & (y == 1))
    fp = sum((preds == 1) & (y == 0))
    fn = sum((preds == 0) & (y == 1))
    tn = sum((preds == 0) & (y == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    api_calls_saved = tn  # Round 2 skip kiye
    print(f"  Threshold={threshold}: precision={precision:.2f}, recall={recall:.2f}, "
          f"API calls saved={api_calls_saved}/{sum(y==0)}")
