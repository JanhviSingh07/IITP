# save_cost_model.py
import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

with open('results/signal_data_combined.json') as f:
    data = json.load(f)

FEATURES = ['cos_sim_top1', 'cos_sim_gap', 'cos_sim_variance',
            'llm_confidence', 'sql_complexity_r1', 'question_length',
            'num_tables_in_db']

X = np.array([[d.get(f, 0) for f in FEATURES] for d in data])
y = np.array([0 if d['r1_correct'] else 1 for d in data])

# Train final model on ALL data (for deployment use)
model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=4, random_state=42)
model.fit(X, y)

# Find threshold that maximizes F1 (via precision-recall curve on training data)
probs = model.predict_proba(X)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y, probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1])  # last point has no threshold
best_threshold = thresholds[best_idx]

print(f"Best threshold: {best_threshold:.3f}")
print(f"At this threshold -> Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

# Save model + threshold + feature order
with open('cost_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'threshold': best_threshold, 'features': FEATURES}, f)

print("Saved cost_model.pkl")