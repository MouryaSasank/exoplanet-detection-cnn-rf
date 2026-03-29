"""
hybrid_classifier.py — CNN Feature Extraction + Random Forest Classification

This module implements the core "hybrid" approach:
  1. Takes a trained CNN and strips off the final classification layers
  2. Uses the penultimate Dense(256) layer to produce a 256-dimensional
     feature vector for each star's light curve
  3. Trains a Random Forest classifier (found via GridSearchCV, then wrapped
     in a sklearn Pipeline with StandardScaler + CalibratedClassifierCV)

Why this works:
  - The CNN learns transit-specific patterns (dips, shapes) from raw flux
  - The RF is robust on tabular data and provides interpretability
  - Combining both gives us deep features + ensemble robustness
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV


def extract_cnn_features(cnn_model, X):
    """
    Extract feature embeddings from the CNN's penultimate layer.

    Uses a sub-model that outputs the activations of the 'feature_layer'
    (Dense-256) instead of the final sigmoid. Each input star gets a
    256-dimensional feature vector.

    Parameters
    ----------
    cnn_model : tf.keras.Model
        The trained (or loaded) CNN model
    X : np.ndarray of shape (n_samples, n_flux, 1)
        Light curve data in CNN-ready 3D format

    Returns
    -------
    features : np.ndarray of shape (n_samples, 256)
        Learned feature vectors
    """
    if not cnn_model.built:
        cnn_model.predict(X[:1], verbose=0)

    feature_layer = cnn_model.get_layer('feature_layer')
    feature_extractor = models.Model(
        inputs=cnn_model.inputs,
        outputs=feature_layer.output,
        name='feature_extractor'
    )

    print("[INFO] Extracting CNN features from the 'feature_layer' (Dense-256)...")
    features = feature_extractor.predict(X, verbose=0)
    print(f"  Extracted feature shape: {features.shape}")
    return features


def train_random_forest(X_features, y, n_estimators=500, random_state=42):
    """
    Use GridSearchCV to find the best RF hyperparameters, then wrap the
    best estimator in a Pipeline with StandardScaler + CalibratedClassifierCV.

    GridSearchCV searches over:
      - classifier__n_estimators : [100, 300, 500]
      - classifier__max_depth    : [10, 20, 30, None]
      - classifier__min_samples_leaf : [1, 2, 4]

    Scoring: f1 (better metric than accuracy for imbalanced data)
    Cross-validation: cv=3 stratified folds

    Parameters
    ----------
    X_features : np.ndarray of shape (n_samples, 256)
        Feature vectors from the CNN
    y : np.ndarray
        Binary labels
    n_estimators : int
        Default fallback (GridSearchCV overrides this)
    random_state : int
        Seed for reproducibility

    Returns
    -------
    final_pipeline : Pipeline
        Trained pipeline (StandardScaler -> CalibratedClassifierCV(best RF))
    """
    print("[INFO] Running GridSearchCV to find best RF hyperparameters...")

    # Step 1: GridSearchCV on a simple Pipeline (Scaler + raw RF, no calibration yet).
    # Calibration is added AFTER finding the best params to avoid CV-within-CV issues.
    search_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    param_grid = {
        'classifier__n_estimators': [300, 500],
        'classifier__max_depth': [20, 30, None],
        'classifier__min_samples_leaf': [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=search_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid_search.fit(X_features, y)

    best_params = grid_search.best_params_
    best_score  = grid_search.best_score_
    print(f"  Best params: {best_params}")
    print(f"  Best CV F1:  {best_score:.4f}")

    # Step 2: Rebuild the best RF with those params, then add probability calibration.
    best_rf = RandomForestClassifier(
        n_estimators=best_params['classifier__n_estimators'],
        max_depth=best_params['classifier__max_depth'],
        min_samples_leaf=best_params['classifier__min_samples_leaf'],
        min_samples_split=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    calibrated_rf = CalibratedClassifierCV(best_rf, cv=3, method='isotonic')

    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', calibrated_rf)
    ])

    final_pipeline.fit(X_features, y)
    print("  Final pipeline trained (StandardScaler -> CalibratedClassifierCV(best RF)).")
    return final_pipeline


def find_optimal_threshold_rf(pipeline, X_val, y_val):
    """
    Find the classification threshold that maximizes F1-score for the RF pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline with predict_proba support
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        True labels

    Returns
    -------
    best_threshold : float
    best_f1 : float
    """
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.10, 0.91, 0.01):
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    if best_f1 == 0:
        best_threshold = 0.2
        print(f"[WARN] No threshold improvement found — using fallback threshold: {best_threshold:.2f}")
    else:
        print(f"[INFO] Optimal RF threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold, best_f1


def save_rf(model, path):
    """Save the trained pipeline to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Hybrid pipeline saved to: {path}")


def load_rf(path):
    """Load a saved pipeline from disk."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] Hybrid pipeline loaded from: {path}")
    return model
