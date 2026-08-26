import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

with open('results/signal_data_combined.json') as f:
    data = json.load(f)

FEATURES = ['cos_sim_top1', 'cos_sim_gap', 'cos_sim_variance',
            'llm_confidence', 'sql_complexity_r1', 'question_length',
            'num_tables_in_db']

X = np.array([[d.get(f, 0) for f in FEATURES] for d in data])
# NEW TARGET: predict whether round-1 will FAIL (this is what should trigger refinement)
y = np.array([0 if d['r1_correct'] else 1 for d in data])

print(f"Total samples: {len(y)}, R1-fail (positive): {y.sum()}, R1-correct (negative): {(y==0).sum()}")

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=4),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=4, random_state=42),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    y_true_all, y_pred_all, y_prob_all = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        y_true_all.extend(y_test)
        y_pred_all.extend(pred)
        y_prob_all.extend(prob)

    y_true_all, y_pred_all, y_prob_all = np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)
    prec = precision_score(y_true_all, y_pred_all, zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    auc = roc_auc_score(y_true_all, y_prob_all)

    print(f"\n{name} (5-fold Stratified CV):")
    print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  AUC-ROC: {auc:.3f}")

# Feature importance
final_lr = LogisticRegression(class_weight='balanced', max_iter=1000)
final_lr.fit(X, y)
print("\n=== Feature coefficients (predicting Round-1 failure) ===")
for feat, coef in sorted(zip(FEATURES, final_lr.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"  {feat}: {coef:.3f}")