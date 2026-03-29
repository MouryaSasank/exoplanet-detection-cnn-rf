"""
preprocessing.py — Data Loading, Normalization, and SMOTE Oversampling

This module handles:
  1. Loading the Kepler exoplanet CSV dataset (train + test splits)
  2. Normalizing each star's flux measurements (zero-mean, unit-variance)
  3. Global feature-wise scaling (fitted on train only, applied to test)
  4. Reshaping data for the 1D CNN input
  5. Applying SMOTE to balance the minority class (confirmed exoplanets)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(train_path, test_path):
    """
    Load the Kepler exoplanet dataset from CSV files.

    The CSV has a 'LABEL' column (2 = exoplanet, 1 = non-exoplanet)
    and ~3197 FLUX columns representing the light curve time-series.

    Parameters
    ----------
    train_path : str
        Path to exoTrain.csv
    test_path : str
        Path to exoTest.csv

    Returns
    -------
    X_train, X_test : np.ndarray
        Flux features arrays of shape (n_samples, n_flux_columns)
    y_train, y_test : np.ndarray
        Binary labels (1 = exoplanet, 0 = non-exoplanet)
    """
    print("[INFO] Loading training data...")
    train_df = pd.read_csv(train_path)
    print(f"  Training set shape: {train_df.shape}")

    print("[INFO] Loading test data...")
    test_df = pd.read_csv(test_path)
    print(f"  Test set shape: {test_df.shape}")

    # Separate labels from features
    # Labels: 2 = CONFIRMED exoplanet, 1 = FALSE POSITIVE (non-exoplanet)
    # We map: 2 -> 1 (exoplanet), 1 -> 0 (non-exoplanet)
    y_train = (train_df['LABEL'] == 2).astype(int).values
    y_test = (test_df['LABEL'] == 2).astype(int).values

    # All columns except 'LABEL' are flux measurements
    X_train = train_df.drop(columns=['LABEL']).values.astype(np.float32)
    X_test = test_df.drop(columns=['LABEL']).values.astype(np.float32)

    print(f"  Training samples: {len(y_train)} | Exoplanets: {y_train.sum()} | Non-exoplanets: {(y_train == 0).sum()}")
    print(f"  Test samples:     {len(y_test)} | Exoplanets: {y_test.sum()} | Non-exoplanets: {(y_test == 0).sum()}")

    return X_train, X_test, y_train, y_test


def normalize_flux(X_train, X_test):
    """
    Normalize flux values using per-sample standardization followed by
    global feature-wise scaling.

    Step 1: Per-sample normalization removes brightness differences between
            stars so the CNN can focus on transit dip shapes.
    Step 2: Global StandardScaler (fitted on train only) ensures each flux
            column has zero mean and unit variance across all training stars,
            which helps the Random Forest downstream.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Raw flux arrays

    Returns
    -------
    X_train_norm, X_test_norm : np.ndarray
        Normalized flux arrays
    scaler : StandardScaler
        The fitted scaler (useful for pipeline consistency)
    """
    print("[INFO] Normalizing flux data...")

    # Step 1: Per-sample normalization (each row independently)
    print("  Step 1: Per-sample standardization...")
    train_mean = X_train.mean(axis=1, keepdims=True)
    train_std = X_train.std(axis=1, keepdims=True) + 1e-8

    test_mean = X_test.mean(axis=1, keepdims=True)
    test_std = X_test.std(axis=1, keepdims=True) + 1e-8

    X_train_ps = (X_train - train_mean) / train_std
    X_test_ps = (X_test - test_mean) / test_std

    # Step 2: Global feature-wise scaling (fit on train only!)
    print("  Step 2: Global feature-wise StandardScaler (fit on train only)...")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_ps).astype(np.float32)
    X_test_norm = scaler.transform(X_test_ps).astype(np.float32)

    print("  Normalization complete.")
    return X_train_norm, X_test_norm, scaler


def reshape_for_cnn(X):
    """
    Reshape 2D flux data (n_samples, n_features) into 3D for Conv1D input.

    Conv1D expects (n_samples, timesteps, channels).
    We treat each flux measurement as one timestep with 1 channel.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    -------
    X_reshaped : np.ndarray of shape (n_samples, n_features, 1)
    """
    return X.reshape(X.shape[0], X.shape[1], 1)


def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.

    The Kepler dataset has severe class imbalance (~37 exoplanets vs ~5050
    non-exoplanets in training). SMOTE creates synthetic examples of the
    minority class by interpolating between existing minority samples.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (2D)
    y : np.ndarray
        Binary labels
    random_state : int
        Seed for reproducibility

    Returns
    -------
    X_resampled, y_resampled : np.ndarray
        Balanced feature matrix and labels
    """
    print("[INFO] Applying SMOTE to handle class imbalance...")
    print(f"  Before SMOTE: Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"  After  SMOTE: Class 0: {(y_resampled == 0).sum()}, Class 1: {(y_resampled == 1).sum()}")
    return X_resampled, y_resampled
