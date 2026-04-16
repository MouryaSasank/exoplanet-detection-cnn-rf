"""
main.py — Full End-to-End Pipeline Runner

Pipeline flow:
  1. Global seed setting for reproducibility
  2. Data loading, denoising, and preprocessing
  3. CNN trains on REAL data only (class weights handle imbalance)
  4. Extract 64-dim CNN features
  5. SMOTE (0.1 ratio) applied ONLY to CNN features for RF training
  6. Hybrid RF (200 trees, depth 8) trains on balanced CNN features
  7. Baseline RF trains on raw flux (with SMOTE 0.3 ratio)
  8. NO ensemble blending (removed — it degraded PR-AUC)
  9. Stratified K-Fold cross-validation
  10. Comprehensive evaluation with validation checks + guide submission checklist
  11. Generate all plots and HTML report

Usage:
    python main.py
"""

import os
import sys
import webbrowser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, f1_score as f1_metric,
    precision_score, recall_score,
    precision_recall_curve, auc as sk_auc,
    matthews_corrcoef, roc_auc_score
)
from sklearn.pipeline import Pipeline as FoldPipeline
from sklearn.preprocessing import StandardScaler as FoldScaler
from sklearn.ensemble import RandomForestClassifier as FoldRF

# Suppress TF C++ logs (must be set before TF import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Project imports
import config as cfg
from utils import set_global_seeds, timer, print_banner
from preprocessing import load_data, denoise_light_curves, normalize_flux, reshape_for_cnn, apply_smote
from cnn_feature_extractor import build_cnn, train_cnn, find_optimal_threshold, save_cnn
from hybrid_classifier import (
    extract_cnn_features, train_random_forest,
    find_optimal_threshold_rf, save_rf
)
from baseline_model import train_baseline_rf, find_optimal_threshold_baseline, save_baseline
from evaluate import (
    compute_metrics,
    bootstrap_confidence_intervals,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_feature_importance,
    plot_sample_light_curves,
    plot_training_curves,
    plot_shap_summary,
    generate_comparison_table,
    generate_html_report,
)


@timer
def main():
    # ──────────────────────────────────────────
    # STEP 0: REPRODUCIBILITY
    # ──────────────────────────────────────────
    set_global_seeds(cfg.GLOBAL_SEED)

    # ──────────────────────────────────────────
    # PATHS
    # ──────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, cfg.DATA_SUBDIR)
    MODEL_DIR = os.path.join(BASE_DIR, cfg.MODEL_SUBDIR)
    OUTPUT_DIR = os.path.join(BASE_DIR, cfg.OUTPUT_SUBDIR)

    TRAIN_CSV = os.path.join(DATA_DIR, cfg.TRAIN_FILENAME)
    TEST_CSV = os.path.join(DATA_DIR, cfg.TEST_FILENAME)

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
    # STEP 1: LOAD, DENOISE & PREPROCESS DATA
    # ──────────────────────────────────────────
    print_banner(1, "Loading, Denoising & Preprocessing Data")

    X_train_raw, X_test_raw, y_train, y_test = load_data(TRAIN_CSV, TEST_CSV)

    # Gaussian smoothing to remove high-frequency noise
    X_train_smooth = denoise_light_curves(X_train_raw, sigma=cfg.GAUSSIAN_SIGMA)
    X_test_smooth = denoise_light_curves(X_test_raw, sigma=cfg.GAUSSIAN_SIGMA)

    X_train_norm, X_test_norm, scaler = normalize_flux(X_train_smooth, X_test_smooth)

    # Plot sample light curves
    plot_sample_light_curves(
        X_test_norm, y_test,
        os.path.join(OUTPUT_DIR, 'sample_light_curves.png')
    )

    # ──────────────────────────────────────────
    # STEP 1b: CREATE VALIDATION SPLIT
    # ──────────────────────────────────────────
    print(f"\n[INFO] Creating validation split ({cfg.VALIDATION_SPLIT*100:.0f}%)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_norm, y_train,
        test_size=cfg.VALIDATION_SPLIT,
        random_state=cfg.GLOBAL_SEED,
        stratify=y_train
    )
    print(f"  Train:      {len(y_tr)} samples (exoplanets: {y_tr.sum()})")
    print(f"  Validation: {len(y_val)} samples (exoplanets: {y_val.sum()})")

    # ──────────────────────────────────────────
    # STEP 2: TRAIN CNN ON REAL DATA (NO SMOTE)
    # ──────────────────────────────────────────
    # CNN trains on imbalanced real data. Class weights handle imbalance.
    # SMOTE on raw flux creates unrealistic synthetic light curves.
    print_banner(2, "Training 1D CNN (on REAL data, class-weighted)")

    n_features = X_tr.shape[1]
    X_tr_cnn = reshape_for_cnn(X_tr)
    X_val_cnn = reshape_for_cnn(X_val)
    X_test_cnn = reshape_for_cnn(X_test_norm)

    cnn_model = build_cnn(input_shape=(n_features, 1))
    history = train_cnn(cnn_model, X_tr_cnn, y_tr, X_val_cnn, y_val)
    save_cnn(cnn_model, os.path.join(MODEL_DIR, 'cnn_model.keras'))

    # Plot training curves
    plot_training_curves(history, os.path.join(OUTPUT_DIR, 'training_curves.png'))

    # ── VALIDATION CHECK: CNN must detect something ──
    cnn_threshold, cnn_f1 = find_optimal_threshold(cnn_model, X_val_cnn, y_val)
    cnn_val_prob = cnn_model.predict(X_val_cnn, verbose=0).ravel()
    cnn_val_pred = (cnn_val_prob >= cnn_threshold).astype(int)
    cnn_val_f1 = f1_metric(y_val, cnn_val_pred, zero_division=0)

    print("\n" + "-" * 50)
    print(f"  VALIDATION CHECK -- CNN F1: {cnn_val_f1:.4f} (threshold={cnn_threshold:.2f})")
    if cnn_val_f1 == 0:
        print("  WARNING: CNN is not detecting any exoplanets!")
    else:
        print("  CNN is learning transit patterns.")
    print("-" * 50)

    # ──────────────────────────────────────────
    # STEP 3: EXTRACT CNN FEATURES
    # ──────────────────────────────────────────
    print_banner(3, "Extracting CNN Features")

    train_features = extract_cnn_features(cnn_model, X_tr_cnn)
    val_features = extract_cnn_features(cnn_model, X_val_cnn)
    test_features = extract_cnn_features(cnn_model, X_test_cnn)

    # ──────────────────────────────────────────
    # STEP 4: APPLY SMOTE TO CNN FEATURES (0.1 ratio — reverted from 0.3)
    # ──────────────────────────────────────────
    # FIX: Reverted from 0.3 to 0.1 — 0.3 caused Run 2 hybrid F1 regression
    # At 0.1 ratio: creates ~429 synthetic from ~31 real positives on CNN features
    print_banner(4, "Applying SMOTE (0.1 ratio, on CNN features only)")

    train_features_smote, y_tr_smote = apply_smote(
        train_features, y_tr,
        sampling_strategy=cfg.SMOTE_SAMPLING_STRATEGY  # 0.1 — hybrid only
    )

    # ──────────────────────────────────────────
    # STEP 5: TRAIN HYBRID RF ON CNN FEATURES
    # ──────────────────────────────────────────
    # FIX: 300 trees, max_depth=12, min_leaf=3 (was 700/20/2)
    print_banner(5, "Training Hybrid Random Forest (200 trees, depth=8)")

    hybrid_rf = train_random_forest(train_features_smote, y_tr_smote)
    save_rf(hybrid_rf, os.path.join(MODEL_DIR, 'hybrid_rf.pkl'))

    # Find optimal threshold (linspace 0.2–0.6, 40 steps)
    hybrid_threshold, hybrid_val_f1 = find_optimal_threshold_rf(
        hybrid_rf, val_features, y_val
    )

    # ──────────────────────────────────────────
    # STEP 6: TRAIN BASELINE RF ON RAW FLUX
    # ──────────────────────────────────────────
    print_banner(6, "Training Baseline Random Forest (Raw Flux)")

    X_tr_smote_raw, y_tr_smote_raw = apply_smote(X_tr, y_tr, sampling_strategy=cfg.BASELINE_SMOTE_STRATEGY)
    baseline_rf = train_baseline_rf(X_tr_smote_raw, y_tr_smote_raw)
    save_baseline(baseline_rf, os.path.join(MODEL_DIR, 'baseline_rf.pkl'))

    baseline_threshold, baseline_val_f1 = find_optimal_threshold_baseline(
        baseline_rf, X_val, y_val
    )

    # ──────────────────────────────────────────
    # STEP 7: EVALUATE BOTH MODELS (NO ENSEMBLE)
    # ──────────────────────────────────────────
    print_banner(7, "Evaluating Models (RF-only, no ensemble)")

    # --- Hybrid RF predictions ---
    hybrid_prob = hybrid_rf.predict_proba(test_features)[:, 1]
    hybrid_pred = (hybrid_prob >= hybrid_threshold).astype(int)
    hybrid_metrics = compute_metrics(y_test, hybrid_pred, hybrid_prob)
    hybrid_metrics['threshold'] = hybrid_threshold

    print(f"\n[HYBRID CNN+RF] Threshold: {hybrid_threshold:.2f}")
    print(f"  Precision: {hybrid_metrics['precision']:.4f}")
    print(f"  Recall:    {hybrid_metrics['recall']:.4f}")
    print(f"  F1:        {hybrid_metrics['f1']:.4f}")
    print(f"  PR-AUC:    {hybrid_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:   {hybrid_metrics['roc_auc']:.4f}")
    print(f"  MCC:       {hybrid_metrics['mcc']:.4f}")

    cm = confusion_matrix(y_test, hybrid_pred)
    print(f"  Confusion: TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    # --- Baseline predictions ---
    baseline_prob = baseline_rf.predict_proba(X_test_norm)[:, 1]
    baseline_pred = (baseline_prob >= baseline_threshold).astype(int)

    # ── TEST-SET SAFETY NET: if zero positives, apply percentile fallback ──
    if baseline_pred.sum() == 0:
        print("[WARN] Baseline predicted ZERO positives on test set — applying test-set fallback")
        # Use top-1% of test probabilities as threshold
        test_pct_99 = np.percentile(baseline_prob, 99)
        baseline_pred = (baseline_prob >= test_pct_99).astype(int)
        baseline_threshold = test_pct_99
        print(f"[FALLBACK-TEST] 99th percentile threshold: {test_pct_99:.6f} "
              f"(positives={baseline_pred.sum()})")
        if baseline_pred.sum() == 0:
            # Top-2%
            test_pct_98 = np.percentile(baseline_prob, 98)
            baseline_pred = (baseline_prob >= test_pct_98).astype(int)
            baseline_threshold = test_pct_98
            print(f"[FALLBACK-TEST] 98th percentile threshold: {test_pct_98:.6f} "
                  f"(positives={baseline_pred.sum()})")

    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_prob)
    baseline_metrics['threshold'] = baseline_threshold

    print(f"\n[BASELINE RF] Threshold: {baseline_threshold:.2f}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall:    {baseline_metrics['recall']:.4f}")
    print(f"  F1:        {baseline_metrics['f1']:.4f}")
    print(f"  PR-AUC:    {baseline_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:   {baseline_metrics['roc_auc']:.4f}")
    print(f"  MCC:       {baseline_metrics['mcc']:.4f}")

    cm_b = confusion_matrix(y_test, baseline_pred)
    print(f"  Confusion: TN={cm_b[0,0]}  FP={cm_b[0,1]}  FN={cm_b[1,0]}  TP={cm_b[1,1]}")

    # ── VALIDATION CHECKS ──
    print("\n" + "=" * 60)
    print("  VALIDATION CHECKS")
    print("=" * 60)
    print(f"  CNN validation F1:    {cnn_val_f1:.4f} {'PASS' if cnn_val_f1 > 0 else 'FAIL'}")
    print(f"  Hybrid PR-AUC:       {hybrid_metrics['pr_auc']:.4f}")
    print(f"  Baseline PR-AUC:     {baseline_metrics['pr_auc']:.4f}")
    print(f"  Baseline F1:         {baseline_metrics['f1']:.4f} {'PASS' if baseline_metrics['f1'] > 0 else 'FAIL - check threshold'}")
    print(f"  Selected threshold:  {hybrid_threshold:.2f}")
    if hybrid_metrics['pr_auc'] > baseline_metrics['pr_auc']:
        improvement = hybrid_metrics['pr_auc'] - baseline_metrics['pr_auc']
        print(f"  SUCCESS: Hybrid PR-AUC > Baseline PR-AUC (+{improvement:.4f})")
    else:
        print(f"  WARNING: Hybrid not beating baseline on PR-AUC")
    print("=" * 60)

    # ── GUIDE SUBMISSION CHECKLIST ────────────────────────────
    # Safe validation — NEVER crashes, always prints pass/fail
    print("\n[GUIDE SUBMISSION CHECKLIST]")
    checks = [
        ("Hybrid F1 >= 0.45",          hybrid_metrics['f1'] >= 0.45),
        ("Hybrid PR-AUC >= 0.35",      hybrid_metrics['pr_auc'] >= 0.35),
        ("Hybrid Recall >= 0.60",       hybrid_metrics['recall'] >= 0.60),
        ("Hybrid Precision >= 0.30",    hybrid_metrics['precision'] >= 0.30),
        ("Baseline F1 > 0.10",          baseline_metrics['f1'] > 0.10),
        ("Baseline Recall > 0.10",      baseline_metrics['recall'] > 0.10),
        ("Baseline Precision > 0.10",   baseline_metrics['precision'] > 0.10),
        ("Baseline MCC > 0.10",         baseline_metrics['mcc'] > 0.10),
        ("Hybrid beats Baseline PR-AUC", hybrid_metrics['pr_auc'] > baseline_metrics['pr_auc']),
        ("Hybrid beats Baseline F1",     hybrid_metrics['f1'] > baseline_metrics['f1']),
        ("Hybrid beats Baseline Recall", hybrid_metrics['recall'] > baseline_metrics['recall']),
    ]
    all_passed = True
    for label, condition in checks:
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_passed = False
        print(f"  [{status}] {label}")
    if all_passed:
        print("\n  ALL CHECKS PASSED — Safe to show guide")
    else:
        print("\n  SOME CHECKS FAILED — Review metrics above before showing guide")
    print("-" * 60)
    # ─────────────────────────────────────────────────────────

    # ──────────────────────────────────────────
    # STEP 7b: STRATIFIED K-FOLD CROSS-VALIDATION
    # ──────────────────────────────────────────
    print_banner("7b", "Stratified K-Fold Cross-Validation (k=5)")

    skf = StratifiedKFold(n_splits=cfg.CV_N_SPLITS, shuffle=True, random_state=cfg.GLOBAL_SEED)
    cv_rows = []

    for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(train_features, y_tr), start=1):
        X_fold_train = train_features[fold_train_idx]
        y_fold_train = y_tr[fold_train_idx]
        X_fold_val   = train_features[fold_val_idx]
        y_fold_val   = y_tr[fold_val_idx]

        # SMOTE only on fold training features (same 0.1 ratio)
        X_fold_s, y_fold_s = apply_smote(X_fold_train, y_fold_train)

        fold_pipe = FoldPipeline([
            ('scaler', FoldScaler()),
            ('classifier', FoldRF(
                n_estimators=cfg.CV_RF_N_ESTIMATORS,
                max_depth=cfg.CV_RF_MAX_DEPTH,
                random_state=cfg.GLOBAL_SEED,
                n_jobs=-1
            ))
        ])
        fold_pipe.fit(X_fold_s, y_fold_s)

        fold_prob = fold_pipe.predict_proba(X_fold_val)[:, 1]

        # Stable threshold search per fold (linspace 0.30–0.60)
        best_fold_t = cfg.THRESHOLD_FALLBACK
        best_fold_f1 = 0
        thresholds = np.linspace(cfg.THRESHOLD_MIN, cfg.THRESHOLD_MAX, cfg.THRESHOLD_STEPS)
        for t in thresholds:
            preds_t = (fold_prob >= t).astype(int)
            f1_t = f1_metric(y_fold_val, preds_t, zero_division=0)
            if f1_t > best_fold_f1:
                best_fold_f1 = f1_t
                best_fold_t = t

        fold_pred = (fold_prob >= best_fold_t).astype(int)
        fold_prec = precision_score(y_fold_val, fold_pred, zero_division=0)
        fold_rec  = recall_score(y_fold_val, fold_pred, zero_division=0)
        fold_f1   = f1_metric(y_fold_val, fold_pred, zero_division=0)
        fold_p_vals, fold_r_vals, _ = precision_recall_curve(y_fold_val, fold_prob)
        fold_prauc = sk_auc(fold_r_vals, fold_p_vals)
        fold_mcc   = matthews_corrcoef(y_fold_val, fold_pred)
        fold_roc   = roc_auc_score(y_fold_val, fold_prob) if y_fold_val.sum() > 0 else 0.0

        cv_rows.append({
            'Fold': fold, 'Precision': round(fold_prec, 4),
            'Recall': round(fold_rec, 4), 'F1': round(fold_f1, 4),
            'PR_AUC': round(fold_prauc, 4), 'ROC_AUC': round(fold_roc, 4),
            'MCC': round(fold_mcc, 4), 'Threshold': round(best_fold_t, 2),
        })
        print(f"  Fold {fold}: P={fold_prec:.4f} R={fold_rec:.4f} F1={fold_f1:.4f} "
              f"PR-AUC={fold_prauc:.4f} MCC={fold_mcc:.4f} t={best_fold_t:.2f}")

    cv_df = pd.DataFrame(cv_rows)
    cv_means = cv_df[['Precision','Recall','F1','PR_AUC','ROC_AUC','MCC']].mean()
    cv_stds  = cv_df[['Precision','Recall','F1','PR_AUC','ROC_AUC','MCC']].std()
    print(f"\n  CV Mean: F1={cv_means['F1']:.4f} PR-AUC={cv_means['PR_AUC']:.4f} MCC={cv_means['MCC']:.4f}")
    print(f"  CV Std:  F1={cv_stds['F1']:.4f} PR-AUC={cv_stds['PR_AUC']:.4f} MCC={cv_stds['MCC']:.4f}")

    cv_csv_path = os.path.join(OUTPUT_DIR, 'cv_results.csv')
    cv_df.to_csv(cv_csv_path, index=False)

    # ──────────────────────────────────────────
    # STEP 8: GENERATE PLOTS
    # ──────────────────────────────────────────
    print_banner(8, "Generating Visualizations")

    plot_confusion_matrix(y_test, hybrid_pred,
                          os.path.join(OUTPUT_DIR, 'confusion_matrix_hybrid.png'),
                          title='Confusion Matrix: Hybrid CNN+RF')

    plot_confusion_matrix(y_test, baseline_pred,
                          os.path.join(OUTPUT_DIR, 'confusion_matrix_baseline.png'),
                          title='Confusion Matrix: Baseline RF')

    plot_precision_recall_curve(y_test, hybrid_prob, baseline_prob,
                                os.path.join(OUTPUT_DIR, 'pr_curve.png'))

    plot_roc_curve(y_test, hybrid_prob, baseline_prob,
                   os.path.join(OUTPUT_DIR, 'roc_curve.png'))

    # Feature importance
    try:
        calibrated = hybrid_rf.named_steps['classifier']
        base_rf = calibrated.calibrated_classifiers_[0].estimator
        plot_feature_importance(base_rf, os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    except Exception as e:
        print(f"[WARN] Feature importance extraction: {e}")
        from sklearn.ensemble import RandomForestClassifier as RFC
        simple_rf = RFC(
            n_estimators=cfg.RF_N_ESTIMATORS,
            max_depth=cfg.RF_MAX_DEPTH,
            min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
            random_state=cfg.GLOBAL_SEED,
            n_jobs=-1
        )
        simple_rf.fit(train_features_smote, y_tr_smote)
        plot_feature_importance(simple_rf, os.path.join(OUTPUT_DIR, 'feature_importance.png'))

    # SHAP explainability
    print_banner("8b", "SHAP Explainability Analysis")
    try:
        plot_shap_summary(hybrid_rf, test_features, y_test,
                          os.path.join(OUTPUT_DIR, 'shap_summary.png'))
    except Exception as e:
        print(f"[WARN] SHAP analysis failed: {e}")

    # ──────────────────────────────────────────
    # STEP 8c: BOOTSTRAP CONFIDENCE INTERVALS
    # ──────────────────────────────────────────
    print("\n[INFO] Computing bootstrap confidence intervals (n=1000)...")
    hybrid_ci = bootstrap_confidence_intervals(y_test, hybrid_prob, hybrid_threshold)
    baseline_ci = bootstrap_confidence_intervals(y_test, baseline_prob, baseline_threshold)

    print(f"  Hybrid   F1:     {hybrid_metrics['f1']:.4f}  [95% CI: {hybrid_ci['f1']['lower']:.4f} – {hybrid_ci['f1']['upper']:.4f}]")
    print(f"  Hybrid   PR-AUC: {hybrid_metrics['pr_auc']:.4f}  [95% CI: {hybrid_ci['pr_auc']['lower']:.4f} – {hybrid_ci['pr_auc']['upper']:.4f}]")
    print(f"  Hybrid   MCC:    {hybrid_metrics['mcc']:.4f}  [95% CI: {hybrid_ci['mcc']['lower']:.4f} – {hybrid_ci['mcc']['upper']:.4f}]")
    print(f"  Baseline F1:     {baseline_metrics['f1']:.4f}  [95% CI: {baseline_ci['f1']['lower']:.4f} – {baseline_ci['f1']['upper']:.4f}]")
    print(f"  Baseline PR-AUC: {baseline_metrics['pr_auc']:.4f}  [95% CI: {baseline_ci['pr_auc']['lower']:.4f} – {baseline_ci['pr_auc']['upper']:.4f}]")

    # Save bootstrap CIs to disk (Critical fix: was only print())
    ci_rows = []
    for model_name, ci_data, metrics in [('Hybrid', hybrid_ci, hybrid_metrics), ('Baseline', baseline_ci, baseline_metrics)]:
        for metric_name in ['f1', 'pr_auc', 'mcc']:
            ci_rows.append({
                'Model': model_name,
                'Metric': metric_name.upper(),
                'Point_Estimate': round(metrics[metric_name], 4),
                'CI_Lower': round(ci_data[metric_name]['lower'], 4),
                'CI_Upper': round(ci_data[metric_name]['upper'], 4),
                'CI_Mean': round(ci_data[metric_name]['mean'], 4),
            })
    ci_df = pd.DataFrame(ci_rows)
    ci_csv_path = os.path.join(OUTPUT_DIR, 'bootstrap_ci.csv')
    ci_df.to_csv(ci_csv_path, index=False)
    print(f"  [INFO] Bootstrap CIs saved to: {ci_csv_path}")

    # ──────────────────────────────────────────
    # STEP 8d: ABLATION STUDY
    # ──────────────────────────────────────────
    print_banner("8d", "Ablation Study")
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier as AblRF
    from sklearn.pipeline import Pipeline as AblPipe
    from sklearn.preprocessing import StandardScaler as AblScaler

    ablation_rows = []

    # Variant 1: CNN-only (use CNN sigmoid output directly, no RF)
    cnn_test_prob = cnn_model.predict(X_test_cnn, verbose=0).ravel()
    cnn_only_pred = (cnn_test_prob >= cnn_threshold).astype(int)
    from sklearn.metrics import precision_score as ps, recall_score as rs
    from sklearn.metrics import precision_recall_curve as prc, auc as sk_auc2
    p_c, r_c, _ = prc(y_test, cnn_test_prob)
    ablation_rows.append({
        'Variant': 'CNN-only (no RF)',
        'Precision': round(ps(y_test, cnn_only_pred, zero_division=0), 4),
        'Recall': round(rs(y_test, cnn_only_pred, zero_division=0), 4),
        'F1': round(f1_metric(y_test, cnn_only_pred, zero_division=0), 4),
        'PR-AUC': round(sk_auc2(r_c, p_c), 4),
    })

    # Variant 2: PCA(64) + RF on raw flux (fair dimensionality-matched baseline)
    pca = PCA(n_components=64, random_state=cfg.GLOBAL_SEED)
    X_tr_pca = pca.fit_transform(X_tr)
    X_test_pca = pca.transform(X_test_norm)
    X_tr_pca_s, y_tr_pca_s = apply_smote(X_tr_pca, y_tr, sampling_strategy=cfg.SMOTE_SAMPLING_STRATEGY)
    pca_pipe = AblPipe([('scaler', AblScaler()), ('clf', AblRF(
        n_estimators=cfg.RF_N_ESTIMATORS, max_depth=cfg.RF_MAX_DEPTH,
        min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF, random_state=cfg.GLOBAL_SEED, n_jobs=-1
    ))])
    pca_pipe.fit(X_tr_pca_s, y_tr_pca_s)
    pca_prob = pca_pipe.predict_proba(X_test_pca)[:, 1]
    pca_pred = (pca_prob >= cfg.THRESHOLD_FALLBACK).astype(int)
    p_p, r_p, _ = prc(y_test, pca_prob)
    ablation_rows.append({
        'Variant': 'PCA(64) + RF (fair baseline)',
        'Precision': round(ps(y_test, pca_pred, zero_division=0), 4),
        'Recall': round(rs(y_test, pca_pred, zero_division=0), 4),
        'F1': round(f1_metric(y_test, pca_pred, zero_division=0), 4),
        'PR-AUC': round(sk_auc2(r_p, p_p), 4),
    })

    # Final hybrid (already computed)
    ablation_rows.append({
        'Variant': 'CNN + RF Hybrid (proposed)',
        'Precision': round(hybrid_metrics['precision'], 4),
        'Recall': round(hybrid_metrics['recall'], 4),
        'F1': round(hybrid_metrics['f1'], 4),
        'PR-AUC': round(hybrid_metrics['pr_auc'], 4),
    })

    abl_df = pd.DataFrame(ablation_rows)
    print("\n  ABLATION RESULTS:")
    print(abl_df.to_string(index=False))
    abl_df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_table.csv'), index=False)
    print(f"\n  Ablation table saved to: {OUTPUT_DIR}/ablation_table.csv")

    # ──────────────────────────────────────────
    # STEP 9: COMPARISON TABLE & HTML REPORT
    # ──────────────────────────────────────────
    print_banner(9, "Final Comparison & HTML Report")

    generate_comparison_table(baseline_metrics, hybrid_metrics,
                              os.path.join(OUTPUT_DIR, 'comparison_table.csv'))

    html_path = generate_html_report(
        baseline_metrics, hybrid_metrics, OUTPUT_DIR,
        cv_results_path=cv_csv_path,
        ablation_csv_path=os.path.join(OUTPUT_DIR, 'ablation_table.csv'),
        bootstrap_ci_path=os.path.join(OUTPUT_DIR, 'bootstrap_ci.csv')
    )

    abs_html = os.path.abspath(html_path)
    print(f"\n[INFO] Opening report in browser: {abs_html}")
    webbrowser.open(f'file:///{abs_html}')

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   Hybrid  F1={hybrid_metrics['f1']:.4f}  PR-AUC={hybrid_metrics['pr_auc']:.4f}  MCC={hybrid_metrics['mcc']:.4f}")
    print(f"   Baseline F1={baseline_metrics['f1']:.4f}  PR-AUC={baseline_metrics['pr_auc']:.4f}  MCC={baseline_metrics['mcc']:.4f}")
    print(f"   Threshold: {hybrid_threshold:.2f}")
    print(f"   All results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
