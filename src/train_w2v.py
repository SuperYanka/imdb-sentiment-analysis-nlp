#  train_w2v.py

import os
import sys
from preprocess import load_and_preprocess_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import joblib
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'data', 'IMDB Dataset.csv')
sys.path.append(os.path.join(BASE_DIR, '..', 'src'))

X_train, X_test, y_train, y_test, w2v_model = load_and_preprocess_text(csv_path)

metrics = {}

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

y_prob = clf.predict_proba(X_test)[:, 1]

acc = round(clf.score(X_test, y_test), 3)
print(f"Accuracy: {acc:.3f}")
auc = round(roc_auc_score(y_test, y_prob), 3)
print(f"ROC AUC: {auc:.3f}")
fpr, tpr, _ = roc_curve(y_test, y_pred)


metrics = {
    "accuracy": acc,
    "roc_auc": auc,
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }
}

# Сохраняем классификатор
model_path = os.path.join(BASE_DIR, '..', 'models', 'w2v_clf.pkl')
joblib.dump(clf, model_path)

# Сохраняем w2v модель
embedding_path = os.path.join(BASE_DIR, '..', 'models', 'w2v_embedding.model')
w2v_model.save(embedding_path)

print(f"\nМодель сохранена в {model_path}")
print(f"Word2Vec сохранён в {embedding_path}")

# Сохраняем метрики
metrics_path = os.path.join(BASE_DIR, '..', 'models', 'w2v_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)