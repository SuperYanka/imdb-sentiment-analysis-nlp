# train_td-idf.py

import os
import sys
from preprocess import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

import joblib
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'data', 'IMDB Dataset.csv')
sys.path.append(os.path.join(BASE_DIR, '..', 'src'))

X_train, X_test, y_train, y_test, tfidf = load_and_preprocess_data(csv_path)


models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'NaiveBayes': MultinomialNB()
}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(f"\n{name} classification report:\n")
    print(classification_report(y_test, y_pred))
    acc = round(model.score(X_test, y_test), 3)
    print(f"Accuracy: {acc:.3f}")
    auc = round(roc_auc_score(y_test, y_prob), 3)
    print(f"ROC AUC: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics[name] = {
        "accuracy": acc,
        "roc_auc": auc,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
        }
    }
 
pkl_path = os.path.join(BASE_DIR, '..', 'models', 'sentiment_model.pkl')
joblib.dump(model, pkl_path)
print(f"\nModel {name} saved to {pkl_path}")

tfidf_path = os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl')
joblib.dump(tfidf, tfidf_path)
print(f"TF-IDF vectorizer saved to {tfidf_path}")

metrics_path = os.path.join(BASE_DIR, '..', 'models', 'td-idf_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f" Metrics saved to {metrics_path}")
