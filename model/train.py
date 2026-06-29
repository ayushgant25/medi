# -*- coding: utf-8 -*-
"""
Model training script for Symptom-to-Disease Predictor.
Trains Random Forest, Naive Bayes, and Decision Tree classifiers,
evaluates them, and saves the best model with its artifacts.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))

TRAIN_CSV  = os.path.join(DATA_DIR, 'training.csv')
TEST_CSV   = os.path.join(DATA_DIR, 'testing.csv')

MODEL_FILE   = os.path.join(MODEL_DIR, 'best_model.pkl')
ENCODER_FILE = os.path.join(MODEL_DIR, 'label_encoder.pkl')
FEATURES_FILE = os.path.join(MODEL_DIR, 'features.pkl')
METRICS_FILE  = os.path.join(MODEL_DIR, 'metrics.json')


def load_and_clean(path):
    """Load CSV, drop duplicates, fill NaN symptoms with 0."""
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    feature_cols = [c for c in df.columns if c != 'prognosis']
    df[feature_cols] = df[feature_cols].fillna(0).astype(int)
    return df


def train_and_evaluate():
    # ── 1. Load data ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Symptom-to-Disease Predictor - Model Training")
    print("=" * 60)

    if not os.path.exists(TRAIN_CSV):
        print(f"\n[ERROR] Dataset not found at {TRAIN_CSV}")
        print("  Run: python data/generate_dataset.py")
        sys.exit(1)

    train_df = load_and_clean(TRAIN_CSV)
    test_df  = load_and_clean(TEST_CSV)

    feature_cols = [c for c in train_df.columns if c != 'prognosis']

    X_train = train_df[feature_cols].values
    y_train_raw = train_df['prognosis'].values
    X_test  = test_df[feature_cols].values
    y_test_raw  = test_df['prognosis'].values

    # ── 2. Encode labels ────────────────────────────────────────────────────
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test  = le.transform(y_test_raw)

    print(f"\nData loaded:")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    print(f"  Features         : {len(feature_cols)}")
    print(f"  Diseases         : {len(le.classes_)}")

    # ── 3. Define models ────────────────────────────────────────────────────
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=42, n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42
        ),
        'Naive Bayes (Bernoulli)': BernoulliNB(alpha=1.0),
    }

    results = {}
    best_name, best_score, best_model = None, -1, None

    print("\n" + "-" * 60)
    for name, model in models.items():
        print(f"\n[Training] {name}")

        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train on full training set
        model.fit(X_train, y_train)

        # Test set evaluation
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"  Test Accuracy:      {test_acc:.4f}")

        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0
        )

        results[name] = {
            'cv_mean': float(cv_scores.mean()),
            'cv_std':  float(cv_scores.std()),
            'test_accuracy': float(test_acc),
            'macro_precision': float(report['macro avg']['precision']),
            'macro_recall':    float(report['macro avg']['recall']),
            'macro_f1':        float(report['macro avg']['f1-score']),
        }

        if test_acc > best_score:
            best_score = test_acc
            best_name  = name
            best_model = model

    # ── 4. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    header = f"{'Model':<30} {'CV Acc':>8} {'Test Acc':>10} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<30} {r['cv_mean']:>8.4f} {r['test_accuracy']:>10.4f} {r['macro_f1']:>8.4f}")

    print(f"\n[BEST] Best model: {best_name} (Test Accuracy: {best_score:.4f})")

    # ── 5. Confusion matrix for best model ──────────────────────────────────
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion matrix saved to metrics.json")

    # ── 6. Save artifacts ───────────────────────────────────────────────────
    joblib.dump(best_model,  MODEL_FILE)
    joblib.dump(le,          ENCODER_FILE)
    joblib.dump(feature_cols, FEATURES_FILE)

    metrics = {
        'best_model': best_name,
        'best_test_accuracy': best_score,
        'all_results': results,
        'diseases': list(le.classes_),
        'n_features': len(feature_cols),
        'confusion_matrix': cm.tolist(),
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts:")
    print(f"  {MODEL_FILE}")
    print(f"  {ENCODER_FILE}")
    print(f"  {FEATURES_FILE}")
    print(f"  {METRICS_FILE}")
    print("\nTraining complete! Run app.py to start the web server.\n")


if __name__ == '__main__':
    train_and_evaluate()
