"""
main.py — Full End-to-End Pipeline Runner

This is the main entry point. It orchestrates:
  1. Data loading and preprocessing (per-sample + global scaling)
  2. CNN training and feature extraction
  3. Hybrid (CNN+RF) model training with SMOTE + threshold optimization
  4. Baseline (raw flux RF) model training with SMOTE + threshold optimization
  5. Evaluation and comparison of both models
  6. Generation of all plots and the HTML report
  7. Opens the report in the default web browser

Usage:
    python main.py
"""

import os
import sys
import webbrowser
import numpy as np
from sklearn.model_selection import train_test_split

# Project imports
from preprocessing import load_data, normalize_flux, reshape_for_cnn, apply_smote
from cnn_feature_extractor import build_cnn, train_cnn, find_optimal_threshold, save_cnn
from hybrid_classifier import extract_cnn_features, train_random_forest, find_optimal_threshold_rf, save_rf
from baseline_model import train_baseline_rf, find_optimal_threshold_baseline, save_baseline
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_sample_light_curves,
    generate_comparison_table,
    generate_html_report,
)


def main():
    # ──────────────────────────────────────────
    # PATHS
    # ──────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

    TRAIN_CSV = os.path.join(DATA_DIR, 'exoTrain.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'exoTest.csv')

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify dataset exists
    if not os.path.isfile(TRAIN_CSV) or not os.path.isfile(TEST_CSV):
        print("=" * 60)
        print("ERROR: Dataset files not found!")
        print(f"  Expected: {TRAIN_CSV}")
        print(f"           {TEST_CSV}")
        print()
        print("Please download the dataset from Kaggle:")
        print("  https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data")
        print("Place exoTrain.csv and exoTest.csv in the 'data/' folder.")
        print("=" * 60)
        sys.exit(1)

    # ──────────────────────────────────────────
    # STEP 1: LOAD & PREPROCESS DATA
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Loading & Preprocessing Data")
    print("=" * 60)

    X_train_raw, X_test_raw, y_train, y_test = load_data(TRAIN_CSV, TEST_CSV)
    X_train_norm, X_test_norm, scaler = normalize_flux(X_train_raw, X_test_raw)

    # Plot sample light curves (from normalized data)
    plot_sample_light_curves(
        X_test_norm, y_test,
        os.path.join(OUTPUT_DIR, 'sample_light_curves.png')
    )

    # ──────────────────────────────────────────
    # STEP 1b: CREATE VALIDATION SPLIT
    # ──────────────────────────────────────────
    # Hold out 15% of training data for threshold optimization
    # Use stratify to preserve class ratios
    print("\n[INFO] Creating validation split (15% of training data, stratified)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_norm, y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )
    print(f"  Train:      {len(y_tr)} samples (exoplanets: {y_tr.sum()})")
    print(f"  Validation: {len(y_val)} samples (exoplanets: {y_val.sum()})")

    # ──────────────────────────────────────────
    # STEP 2: TRAIN CNN
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Training 1D CNN")
    print("=" * 60)

    n_features = X_tr.shape[1]
    X_tr_cnn = reshape_for_cnn(X_tr)
    X_val_cnn = reshape_for_cnn(X_val)
    X_test_cnn = reshape_for_cnn(X_test_norm)

    cnn_model = build_cnn(input_shape=(n_features, 1))
    history = train_cnn(cnn_model, X_tr_cnn, y_tr, X_val_cnn, y_val,
                        batch_size=32)
    save_cnn(cnn_model, os.path.join(MODEL_DIR, 'cnn_model.keras'))

    # Find optimal CNN threshold on validation data
    cnn_threshold, cnn_f1 = find_optimal_threshold(cnn_model, X_val_cnn, y_val)

    # ──────────────────────────────────────────
    # STEP 3: EXTRACT CNN FEATURES
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Extracting CNN Features")
    print("=" * 60)

    train_features = extract_cnn_features(cnn_model, X_tr_cnn)
    val_features = extract_cnn_features(cnn_model, X_val_cnn)
    test_features = extract_cnn_features(cnn_model, X_test_cnn)

    # ──────────────────────────────────────────
    # STEP 4: APPLY SMOTE TO CNN FEATURES
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Applying SMOTE (on CNN features, training set only)")
    print("=" * 60)

    train_features_smote, y_tr_smote = apply_smote(train_features, y_tr)

    # ──────────────────────────────────────────
    # STEP 5: TRAIN HYBRID RF ON CNN FEATURES
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: Training Hybrid Random Forest (Pipeline + Calibration)")
    print("=" * 60)

    hybrid_rf = train_random_forest(train_features_smote, y_tr_smote, n_estimators=500)
    save_rf(hybrid_rf, os.path.join(MODEL_DIR, 'hybrid_rf.pkl'))

    # Find optimal threshold on validation CNN features
    hybrid_threshold, hybrid_val_f1 = find_optimal_threshold_rf(
        hybrid_rf, val_features, y_val
    )

    # ──────────────────────────────────────────
    # STEP 6: TRAIN BASELINE RF ON RAW FLUX
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6: Training Baseline Random Forest (Raw Flux)")
    print("=" * 60)

    X_tr_smote_raw, y_tr_smote_raw = apply_smote(X_tr, y_tr)
    baseline_rf = train_baseline_rf(X_tr_smote_raw, y_tr_smote_raw, n_estimators=200)
    save_baseline(baseline_rf, os.path.join(MODEL_DIR, 'baseline_rf.pkl'))

    # Find optimal baseline threshold
    baseline_threshold, baseline_val_f1 = find_optimal_threshold_baseline(
        baseline_rf, X_val, y_val
    )

    # ──────────────────────────────────────────
    # STEP 7: EVALUATE BOTH MODELS
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7: Evaluating Models (with optimized thresholds)")
    print("=" * 60)

    # Hybrid predictions with optimized threshold
    hybrid_prob = hybrid_rf.predict_proba(test_features)[:, 1]
    hybrid_pred = (hybrid_prob >= hybrid_threshold).astype(int)
    hybrid_metrics = compute_metrics(y_test, hybrid_pred, hybrid_prob)
    hybrid_metrics['threshold'] = hybrid_threshold

    print(f"\n[HYBRID CNN+RF] Threshold: {hybrid_threshold:.2f}")
    print("[HYBRID CNN+RF] Classification Report:")
    print(hybrid_metrics['report'])
    print(f"[HYBRID CNN+RF] PR-AUC: {hybrid_metrics['pr_auc']:.4f}")

    # Confusion matrix details
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, hybrid_pred)
    print(f"[HYBRID CNN+RF] Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Baseline predictions with optimized threshold
    baseline_prob = baseline_rf.predict_proba(X_test_norm)[:, 1]
    baseline_pred = (baseline_prob >= baseline_threshold).astype(int)
    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_prob)
    baseline_metrics['threshold'] = baseline_threshold

    print(f"\n[BASELINE RF] Threshold: {baseline_threshold:.2f}")
    print("[BASELINE RF] Classification Report:")
    print(baseline_metrics['report'])
    print(f"[BASELINE RF] PR-AUC: {baseline_metrics['pr_auc']:.4f}")

    cm_b = confusion_matrix(y_test, baseline_pred)
    print(f"[BASELINE RF] Confusion Matrix:")
    print(f"  TN={cm_b[0,0]}  FP={cm_b[0,1]}")
    print(f"  FN={cm_b[1,0]}  TP={cm_b[1,1]}")

    # ──────────────────────────────────────────
    # STEP 7b: STRATIFIED K-FOLD CROSS-VALIDATION
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7b: Stratified K-Fold Cross-Validation (k=5)")
    print("=" * 60)
    print("[INFO] Running 5-fold stratified CV on CNN features (training set only)")

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score as f1_metric
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import precision_recall_curve, auc as pr_auc_score
    from sklearn.pipeline import Pipeline as FoldPipeline
    from sklearn.preprocessing import StandardScaler as FoldScaler
    from sklearn.ensemble import RandomForestClassifier as FoldRF
    import pandas as pd

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []

    for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(train_features, y_tr), start=1):
        X_fold_train = train_features[fold_train_idx]
        y_fold_train = y_tr[fold_train_idx]
        X_fold_val   = train_features[fold_val_idx]
        y_fold_val   = y_tr[fold_val_idx]

        # Balance only the fold training set with SMOTE
        X_fold_s, y_fold_s = apply_smote(X_fold_train, y_fold_train)

        # Lean RF per fold (skip GridSearch to keep CV runtime reasonable)
        fold_pipe = FoldPipeline([
            ('scaler', FoldScaler()),
            ('classifier', FoldRF(
                n_estimators=200,
                class_weight='balanced',
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ))
        ])
        fold_pipe.fit(X_fold_s, y_fold_s)

        fold_prob = fold_pipe.predict_proba(X_fold_val)[:, 1]
        fold_pred = (fold_prob >= 0.3).astype(int)

        fold_prec = precision_score(y_fold_val, fold_pred, zero_division=0)
        fold_rec  = recall_score(y_fold_val, fold_pred, zero_division=0)
        fold_f1   = f1_metric(y_fold_val, fold_pred, zero_division=0)
        fold_p_vals, fold_r_vals, _ = precision_recall_curve(y_fold_val, fold_prob)
        fold_prauc = pr_auc_score(fold_r_vals, fold_p_vals)

        cv_rows.append({
            'Fold': fold,
            'Precision': round(fold_prec, 4),
            'Recall': round(fold_rec, 4),
            'F1': round(fold_f1, 4),
            'PR_AUC': round(fold_prauc, 4),
        })
        print(f"  Fold {fold}: Precision={fold_prec:.4f}  Recall={fold_rec:.4f}  F1={fold_f1:.4f}  PR-AUC={fold_prauc:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    cv_means = cv_df[['Precision','Recall','F1','PR_AUC']].mean()
    cv_stds  = cv_df[['Precision','Recall','F1','PR_AUC']].std()

    print(f"\n  CV Mean: Precision={cv_means['Precision']:.4f}  Recall={cv_means['Recall']:.4f}  F1={cv_means['F1']:.4f}  PR-AUC={cv_means['PR_AUC']:.4f}")
    print(f"  CV Std : Precision={cv_stds['Precision']:.4f}  Recall={cv_stds['Recall']:.4f}  F1={cv_stds['F1']:.4f}  PR-AUC={cv_stds['PR_AUC']:.4f}")

    # Save K-Fold results to CSV
    cv_csv_path = os.path.join(OUTPUT_DIR, 'cv_results.csv')
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"  [INFO] CV results saved to: {cv_csv_path}")

    # ──────────────────────────────────────────
    # STEP 8: GENERATE PLOTS & COMPARISON
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 8: Generating Visualizations")
    print("=" * 60)

    plot_confusion_matrix(y_test, hybrid_pred,
                          os.path.join(OUTPUT_DIR, 'confusion_matrix_hybrid.png'),
                          title='Confusion Matrix: Hybrid CNN+RF')

    plot_confusion_matrix(y_test, baseline_pred,
                          os.path.join(OUTPUT_DIR, 'confusion_matrix_baseline.png'),
                          title='Confusion Matrix: Baseline RF')

    plot_precision_recall_curve(y_test, hybrid_prob, baseline_prob,
                                os.path.join(OUTPUT_DIR, 'pr_curve.png'))

    # For feature importance, extract the base RF from the pipeline
    try:
        # Pipeline -> CalibratedClassifierCV -> fitted RF from one calibrated fold
        calibrated = hybrid_rf.named_steps['classifier']
        base_rf = calibrated.calibrated_classifiers_[0].estimator
        plot_feature_importance(base_rf,
                                os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    except Exception as e:
        print(f"[WARN] Could not plot feature importance: {e}")
        # Fallback: train a simple RF just for feature importance visualization
        from sklearn.ensemble import RandomForestClassifier as RFC
        simple_rf = RFC(n_estimators=100, random_state=42, n_jobs=-1)
        simple_rf.fit(train_features_smote, y_tr_smote)
        plot_feature_importance(simple_rf,
                                os.path.join(OUTPUT_DIR, 'feature_importance.png'))

    # ──────────────────────────────────────────
    # STEP 9: COMPARISON TABLE & HTML REPORT
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 9: Final Comparison & HTML Report")
    print("=" * 60)

    generate_comparison_table(baseline_metrics, hybrid_metrics,
                              os.path.join(OUTPUT_DIR, 'comparison_table.csv'))

    html_path = generate_html_report(
        baseline_metrics, hybrid_metrics, OUTPUT_DIR,
        cv_results_path=os.path.join(OUTPUT_DIR, 'cv_results.csv')
    )

    # Open report in the default browser
    abs_html = os.path.abspath(html_path)
    print(f"\n[INFO] Opening report in browser: {abs_html}")
    webbrowser.open(f'file:///{abs_html}')

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   Hybrid threshold:   {hybrid_threshold:.2f}")
    print(f"   Baseline threshold: {baseline_threshold:.2f}")
    print(f"   All results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
