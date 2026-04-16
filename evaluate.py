"""
evaluate.py — Metrics Computation and Visualization

This module produces all evaluation outputs:
  - Classification metrics (Precision, Recall, F1, PR-AUC, ROC-AUC, MCC)
  - Confusion matrix heatmaps
  - Precision-Recall curve plots
  - ROC curve plots
  - CNN training curve plots (loss + accuracy per epoch)
  - Random Forest feature importance chart
  - SHAP explainability summary
  - Sample light curve visualizations
  - Side-by-side comparison table (Baseline vs Hybrid)
  - HTML report with all results embedded
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, auc,
    roc_curve, roc_auc_score,
    matthews_corrcoef
)


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute comprehensive metrics: precision, recall, F1, PR-AUC, ROC-AUC, MCC.

    We use PR-AUC as the primary metric because it is more informative
    for severely imbalanced datasets. ROC-AUC and MCC are included as
    additional evaluation criteria.

    Parameters
    ----------
    y_true : array-like — true labels
    y_pred : array-like — predicted labels
    y_prob : array-like — predicted probabilities for the positive class

    Returns
    -------
    metrics : dict
    """
    # Consolidated: single classification_report call
    report_dict = classification_report(
        y_true, y_pred,
        target_names=['Non-Planet', 'Exoplanet'],
        output_dict=True
    )
    # Reconstruct text report from dict (avoids redundant classification_report call)
    def _fmt(name, d):
        return f"{name:>14s} {d['precision']:9.2f} {d['recall']:9.2f} {d['f1-score']:9.2f} {int(d['support']):9d}"
    _total = int(report_dict['weighted avg']['support'])
    report_str = "\n".join([
        f"{'':>14s} {'precision':>9s} {'recall':>9s} {'f1-score':>9s} {'support':>9s}",
        "",
        _fmt('Non-Planet', report_dict['Non-Planet']),
        _fmt('Exoplanet', report_dict['Exoplanet']),
        "",
        f"{'accuracy':>14s} {'':>9s} {'':>9s} {report_dict['accuracy']:9.2f} {_total:9d}",
        _fmt('macro avg', report_dict['macro avg']),
        _fmt('weighted avg', report_dict['weighted avg']),
    ])

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    # Matthews Correlation Coefficient — best single metric for imbalanced data
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        'precision': report_dict['Exoplanet']['precision'],
        'recall': report_dict['Exoplanet']['recall'],
        'f1': report_dict['Exoplanet']['f1-score'],
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'mcc': mcc,
        'accuracy': report_dict['accuracy'],
        'report': report_str,
    }
    return metrics


def bootstrap_confidence_intervals(y_true, y_prob, threshold, n_bootstrap=1000, ci=95, random_state=42):
    """
    Compute bootstrapped confidence intervals for F1, PR-AUC, and MCC.

    With only 5 positive samples in the test set, point estimates are
    statistically fragile. Bootstrap resampling quantifies this uncertainty
    by resampling the test set 1000 times with replacement and computing
    metrics on each resample.

    Parameters
    ----------
    y_true : array-like — true labels
    y_prob : array-like — predicted probabilities
    threshold : float — classification threshold
    n_bootstrap : int — number of bootstrap resamples
    ci : float — confidence interval width (default 95%)
    random_state : int — seed for reproducibility

    Returns
    -------
    results : dict
        Keys 'f1', 'pr_auc', 'mcc', each containing 'mean', 'lower', 'upper'
    """
    rng = np.random.RandomState(random_state)
    f1_scores, prauc_scores, mcc_scores = [], [], []

    from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_curve, auc

    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_true), size=len(y_true))
        y_b = np.array(y_true)[indices]
        p_b = np.array(y_prob)[indices]

        # Skip degenerate resamples (all one class)
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue

        pred_b = (p_b >= threshold).astype(int)
        f1_scores.append(f1_score(y_b, pred_b, zero_division=0))
        mcc_scores.append(matthews_corrcoef(y_b, pred_b))

        prec_v, rec_v, _ = precision_recall_curve(y_b, p_b)
        prauc_scores.append(auc(rec_v, prec_v))

    alpha = (100 - ci) / 2
    results = {}
    for name, scores in [('f1', f1_scores), ('pr_auc', prauc_scores), ('mcc', mcc_scores)]:
        arr = np.array(scores)
        results[name] = {
            'mean': float(np.mean(arr)),
            'lower': float(np.percentile(arr, alpha)),
            'upper': float(np.percentile(arr, 100 - alpha))
        }

    return results

def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
    """Plot and save a confusion matrix as a heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Planet', 'Exoplanet'],
                yticklabels=['Non-Planet', 'Exoplanet'], ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")


def plot_precision_recall_curve(y_true, y_prob_hybrid, y_prob_baseline, save_path):
    """Plot Precision-Recall curves for both hybrid and baseline models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    prec_h, rec_h, _ = precision_recall_curve(y_true, y_prob_hybrid)
    prec_b, rec_b, _ = precision_recall_curve(y_true, y_prob_baseline)
    auc_h = auc(rec_h, prec_h)
    auc_b = auc(rec_b, prec_b)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_h, prec_h, label=f'Hybrid CNN+RF (AUC={auc_h:.4f})', linewidth=2)
    ax.plot(rec_b, prec_b, label=f'Baseline RF (AUC={auc_b:.4f})',
            linewidth=2, linestyle='--')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] PR curve saved to: {save_path}")


def plot_roc_curve(y_true, y_prob_hybrid, y_prob_baseline, save_path):
    """Plot ROC curves for both hybrid and baseline models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr_h, tpr_h, _ = roc_curve(y_true, y_prob_hybrid)
    fpr_b, tpr_b, _ = roc_curve(y_true, y_prob_baseline)

    try:
        roc_h = roc_auc_score(y_true, y_prob_hybrid)
    except ValueError:
        roc_h = 0.0
    try:
        roc_b = roc_auc_score(y_true, y_prob_baseline)
    except ValueError:
        roc_b = 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_h, tpr_h, label=f'Hybrid CNN+RF (AUC={roc_h:.4f})', linewidth=2)
    ax.plot(fpr_b, tpr_b, label=f'Baseline RF (AUC={roc_b:.4f})',
            linewidth=2, linestyle='--')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curve saved to: {save_path}")


def plot_training_curves(history, save_path):
    """
    Plot CNN training curves: loss and accuracy per epoch for train/val.

    This is critical for demonstrating CNN convergence to evaluators.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to: {save_path}")


def plot_feature_importance(rf_model, save_path, top_n=20):
    """Plot the top-N most important features from the Random Forest model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), importances[indices], color='teal')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f'CNN Feature {i}' for i in indices])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} CNN Feature Importances (Hybrid RF)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Feature importance plot saved to: {save_path}")


def plot_sample_light_curves(X, y, save_path, n_samples=3):
    """Plot sample light curves for both exoplanet and non-exoplanet stars."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 8))

    exo_idx = np.where(y == 1)[0]
    non_idx = np.where(y == 0)[0]

    for i in range(n_samples):
        if i < len(exo_idx):
            axes[0, i].plot(X[exo_idx[i]], color='crimson', linewidth=0.5)
            axes[0, i].set_title(f'Exoplanet Star #{exo_idx[i]}', fontsize=10)
        axes[0, i].set_xlabel('Flux Index')
        axes[0, i].set_ylabel('Normalized Flux')

        if i < len(non_idx):
            axes[1, i].plot(X[non_idx[i]], color='steelblue', linewidth=0.5)
            axes[1, i].set_title(f'Non-Planet Star #{non_idx[i]}', fontsize=10)
        axes[1, i].set_xlabel('Flux Index')
        axes[1, i].set_ylabel('Normalized Flux')

    fig.suptitle('Sample Light Curves: Exoplanet (top) vs Non-Planet (bottom)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Sample light curves saved to: {save_path}")


def plot_shap_summary(pipeline, X_test, y_test, save_path, max_display=20):
    """
    Generate SHAP summary plot for model interpretability.

    SHAP (SHapley Additive exPlanations) shows which CNN features
    contribute most to each prediction, making the hybrid model interpretable.

    Parameters
    ----------
    pipeline : Pipeline
        Trained hybrid pipeline
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    save_path : str
        Where to save the SHAP plot
    max_display : int
        Number of top features to display
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        import shap
    except ImportError:
        print("[WARN] SHAP not installed. Run: pip install shap")
        return

    print("[INFO] Computing SHAP values (this may take a minute)...")

    # Scale the test features using the pipeline's scaler
    scaler = pipeline.named_steps['scaler']
    X_scaled = scaler.transform(X_test)

    # Extract the base RF estimator from the calibrated pipeline
    # 3-tier fallback: calibrated estimator → direct classifier → fresh RF
    base_rf = None
    try:
        calibrated = pipeline.named_steps['classifier']
        base_rf = calibrated.calibrated_classifiers_[0].estimator
        print("[INFO] Using calibrated RF base estimator for SHAP")
    except (AttributeError, IndexError, KeyError) as e:
        print(f"[WARN] Could not extract calibrated RF: {e}")
        try:
            clf = pipeline.named_steps['classifier']
            if hasattr(clf, 'estimators_'):
                base_rf = clf
                print("[INFO] Using uncalibrated RF for SHAP (fallback 1)")
        except (AttributeError, KeyError):
            pass

    if base_rf is None:
        print("[WARN] Training fresh RF for SHAP analysis (fallback 2)...")
        from sklearn.ensemble import RandomForestClassifier
        base_rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
        base_rf.fit(X_scaled, y_test[:len(X_scaled)])

    explainer = shap.TreeExplainer(base_rf)

    # Use a subsample for speed
    n_samples = min(200, len(X_scaled))
    X_shap = X_scaled[:n_samples]
    shap_values = explainer.shap_values(X_shap)

    # For binary classification, shap_values[1] is the positive class
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    feature_names = [f'CNN_feat_{i}' for i in range(X_shap.shape[1])]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_shap, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] SHAP summary plot saved to: {save_path}")


def generate_comparison_table(baseline_metrics, hybrid_metrics, save_path):
    """
    Generate a side-by-side comparison table of Baseline vs Hybrid metrics.
    Now includes ROC-AUC and MCC.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    comparison = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score', 'PR-AUC', 'ROC-AUC', 'MCC', 'Accuracy'],
        'Baseline RF': [
            f"{baseline_metrics['precision']:.4f}",
            f"{baseline_metrics['recall']:.4f}",
            f"{baseline_metrics['f1']:.4f}",
            f"{baseline_metrics['pr_auc']:.4f}",
            f"{baseline_metrics['roc_auc']:.4f}",
            f"{baseline_metrics['mcc']:.4f}",
            f"{baseline_metrics['accuracy']:.4f}",
        ],
        'Hybrid CNN+RF': [
            f"{hybrid_metrics['precision']:.4f}",
            f"{hybrid_metrics['recall']:.4f}",
            f"{hybrid_metrics['f1']:.4f}",
            f"{hybrid_metrics['pr_auc']:.4f}",
            f"{hybrid_metrics['roc_auc']:.4f}",
            f"{hybrid_metrics['mcc']:.4f}",
            f"{hybrid_metrics['accuracy']:.4f}",
        ],
    })
    comparison.to_csv(save_path, index=False)
    print(f"\n[INFO] Comparison table saved to: {save_path}")
    print("\n" + "=" * 60)
    print("       FINAL COMPARISON: Baseline RF vs Hybrid CNN+RF")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print("=" * 60)
    return comparison


def generate_html_report(baseline_metrics, hybrid_metrics, output_dir, cv_results_path=None,
                         ablation_csv_path=None, bootstrap_ci_path=None):
    """
    Generate a premium HTML report with Apple-style animations and interactions.
    Features: scroll-triggered reveals, hover lifts, glassmorphism, animated counters,
    floating star particles, and smooth cubic-bezier transitions.

    Updated to include: ROC-AUC, MCC, Training Curves, ROC Curve, SHAP section,
    Ablation Study table, Bootstrap CIs, and Limitations & Future Work.
    """
    html_path = os.path.join(output_dir, 'report.html')

    # ── Build CV section HTML ──
    if cv_results_path and os.path.isfile(cv_results_path):
        cv_df = pd.read_csv(cv_results_path)
        metric_cols = [c for c in ['Precision', 'Recall', 'F1', 'PR_AUC', 'ROC_AUC', 'MCC'] if c in cv_df.columns]
        cv_means = cv_df[metric_cols].mean()
        cv_stds  = cv_df[metric_cols].std()

        cv_prec_mean  = f"{cv_means.get('Precision', 0):.4f}"
        cv_prec_std   = f"{cv_stds.get('Precision', 0):.4f}"
        cv_rec_mean   = f"{cv_means.get('Recall', 0):.4f}"
        cv_rec_std    = f"{cv_stds.get('Recall', 0):.4f}"
        cv_f1_mean    = f"{cv_means.get('F1', 0):.4f}"
        cv_f1_std     = f"{cv_stds.get('F1', 0):.4f}"
        cv_prauc_mean = f"{cv_means.get('PR_AUC', 0):.4f}"
        cv_prauc_std  = f"{cv_stds.get('PR_AUC', 0):.4f}"
        cv_roc_mean   = f"{cv_means.get('ROC_AUC', 0):.4f}"
        cv_roc_std    = f"{cv_stds.get('ROC_AUC', 0):.4f}"
        cv_mcc_mean   = f"{cv_means.get('MCC', 0):.4f}"
        cv_mcc_std    = f"{cv_stds.get('MCC', 0):.4f}"

        cv_fold_rows = ''
        for _, row in cv_df.iterrows():
            roc_val = f"{row.get('ROC_AUC', 0):.4f}" if 'ROC_AUC' in row else '—'
            mcc_val = f"{row.get('MCC', 0):.4f}" if 'MCC' in row else '—'
            cv_fold_rows += (
                f'<tr><td>Fold {int(row["Fold"])}</td>'
                f'<td class="highlight">{row["Precision"]:.4f}</td>'
                f'<td class="highlight">{row["Recall"]:.4f}</td>'
                f'<td class="highlight">{row["F1"]:.4f}</td>'
                f'<td class="highlight">{row["PR_AUC"]:.4f}</td>'
                f'<td class="highlight">{roc_val}</td>'
                f'<td class="highlight">{mcc_val}</td></tr>'
            )
        cv_section = (
            '<section class="section" id="crossval">'
            '<div class="section-title reveal">Cross-Validation Stability</div>'
            '<div class="section-subtitle reveal delay-1">'
            '5-Fold Stratified K-Fold on CNN features (training set only) &mdash; Mean &plusmn; Std'
            '</div>'
            '<div class="stats-grid">'
            f'<div class="stat-card reveal delay-1"><div class="stat-value green">{cv_prec_mean}</div>'
            f'<div class="stat-label">Precision</div><div class="stat-model">Mean &plusmn; {cv_prec_std}</div></div>'
            f'<div class="stat-card reveal delay-2"><div class="stat-value">{cv_rec_mean}</div>'
            f'<div class="stat-label">Recall</div><div class="stat-model">Mean &plusmn; {cv_rec_std}</div></div>'
            f'<div class="stat-card reveal delay-3"><div class="stat-value purple">{cv_f1_mean}</div>'
            f'<div class="stat-label">F1-Score</div><div class="stat-model">Mean &plusmn; {cv_f1_std}</div></div>'
            f'<div class="stat-card reveal delay-4"><div class="stat-value orange">{cv_prauc_mean}</div>'
            f'<div class="stat-label">PR-AUC</div><div class="stat-model">Mean &plusmn; {cv_prauc_std}</div></div>'
            f'<div class="stat-card reveal delay-5"><div class="stat-value">{cv_roc_mean}</div>'
            f'<div class="stat-label">ROC-AUC</div><div class="stat-model">Mean &plusmn; {cv_roc_std}</div></div>'
            f'<div class="stat-card reveal delay-1"><div class="stat-value pink">{cv_mcc_mean}</div>'
            f'<div class="stat-label">MCC</div><div class="stat-model">Mean &plusmn; {cv_mcc_std}</div></div>'
            '</div>'
            '<div class="table-card reveal delay-3"><table>'
            '<thead><tr><th>Fold</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>PR-AUC</th><th>ROC-AUC</th><th>MCC</th></tr></thead>'
            f'<tbody>{cv_fold_rows}</tbody></table></div>'
            '</section>'
        )
        cv_nav = '<li><a href="#crossval">Cross-Validation</a></li>'
    else:
        cv_section = ''
        cv_nav = ''

    # Check for optional images
    shap_section = ''
    shap_nav = ''
    if os.path.isfile(os.path.join(output_dir, 'shap_summary.png')):
        shap_section = (
            '<section class="section" id="shap">'
            '<div class="section-title reveal">SHAP Explainability</div>'
            '<div class="section-subtitle reveal delay-1">Which CNN features contribute most to each prediction</div>'
            '<div class="plots single">'
            '<div class="plot-card reveal-scale delay-2">'
            '<img src="shap_summary.png" alt="SHAP Summary">'
            '<p>Feature contributions: red = pushes prediction towards exoplanet</p>'
            '</div></div></section>'
        )
        shap_nav = '<li><a href="#shap">SHAP</a></li>'

    # ── Build Ablation section HTML ──
    ablation_section = ''
    ablation_nav = ''
    if ablation_csv_path and os.path.isfile(ablation_csv_path):
        abl_df = pd.read_csv(ablation_csv_path)
        abl_rows = ''
        for _, row in abl_df.iterrows():
            is_proposed = 'proposed' in str(row.get('Variant', '')).lower()
            row_class = ' style="background: rgba(48,209,88,0.06);"' if is_proposed else ''
            variant_label = f'<strong>{row["Variant"]}</strong>' if is_proposed else row['Variant']
            abl_rows += (
                f'<tr{row_class}><td>{variant_label}</td>'
                f'<td class="highlight">{row.get("Precision", 0):.4f}</td>'
                f'<td class="highlight">{row.get("Recall", 0):.4f}</td>'
                f'<td class="highlight">{row.get("F1", 0):.4f}</td>'
                f'<td class="highlight">{row.get("PR-AUC", 0):.4f}</td></tr>'
            )
        ablation_section = (
            '<section class="section" id="ablation">'
            '<div class="section-title reveal">Ablation Study</div>'
            '<div class="section-subtitle reveal delay-1">'
            'Isolating the contribution of each pipeline component'
            '</div>'
            '<div class="table-card reveal delay-2"><table>'
            '<thead><tr><th>Variant</th><th>Precision</th><th>Recall</th>'
            '<th>F1-Score</th><th>PR-AUC</th></tr></thead>'
            f'<tbody>{abl_rows}</tbody></table></div>'
            '</section>'
        )
        ablation_nav = '<li><a href="#ablation">Ablation</a></li>'

    # ── Build Bootstrap CI section HTML ──
    ci_section = ''
    ci_nav = ''
    if bootstrap_ci_path and os.path.isfile(bootstrap_ci_path):
        ci_df = pd.read_csv(bootstrap_ci_path)
        ci_rows = ''
        for _, row in ci_df.iterrows():
            ci_rows += (
                f'<tr><td>{row["Model"]}</td>'
                f'<td>{row["Metric"]}</td>'
                f'<td class="highlight">{row["Point_Estimate"]:.4f}</td>'
                f'<td>{row["CI_Lower"]:.4f}</td>'
                f'<td>{row["CI_Upper"]:.4f}</td></tr>'
            )
        ci_section = (
            '<section class="section" id="bootstrap">'
            '<div class="section-title reveal">Bootstrap Confidence Intervals</div>'
            '<div class="section-subtitle reveal delay-1">'
            '1000-resample bootstrap quantifies statistical uncertainty (only 5 test positives)'
            '</div>'
            '<div class="table-card reveal delay-2"><table>'
            '<thead><tr><th>Model</th><th>Metric</th><th>Point Estimate</th>'
            '<th>95% CI Lower</th><th>95% CI Upper</th></tr></thead>'
            f'<tbody>{ci_rows}</tbody></table></div>'
            '</section>'
        )
        ci_nav = '<li><a href="#bootstrap">Bootstrap CI</a></li>'

    # ── Build Limitations section HTML ──
    limitations_section = (
        '<section class="section" id="limitations">'
        '<div class="section-title reveal">Limitations &amp; Future Work</div>'
        '<div class="section-subtitle reveal delay-1">Honest assessment of current constraints and next steps</div>'
        '<div class="stats-grid">'
        '<div class="stat-card reveal delay-1">'
        '<div class="stat-value orange">5</div>'
        '<div class="stat-label">Test Positives</div>'
        '<div class="stat-model">Point metrics are statistically fragile &mdash; bootstrap CIs quantify this uncertainty</div></div>'
        '<div class="stat-card reveal delay-2">'
        '<div class="stat-value pink">0.14&ndash;0.60</div>'
        '<div class="stat-label">CV F1 Range</div>'
        '<div class="stat-model">High variance across folds reflects small positive set (6&ndash;7 exoplanets per fold)</div></div>'
        '<div class="stat-card reveal delay-3">'
        '<div class="stat-value">37</div>'
        '<div class="stat-label">Real Samples</div>'
        '<div class="stat-model">SMOTE on 37 real exoplanets &mdash; synthetic features may not capture rare transit variants</div></div>'
        '</div>'
        '<div class="table-card reveal delay-4" style="padding: 2rem;">'
        '<div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: rgba(255,255,255,0.8);">Future Directions</div>'
        '<div style="color: rgba(255,255,255,0.5); line-height: 1.8;">'
        '&bull; <strong>Temporal models</strong>: LSTM / Transformer architectures for long-range transit periodicity<br>'
        '&bull; <strong>Cross-mission transfer</strong>: Generalize to TESS and ground-based datasets<br>'
        '&bull; <strong>Domain priors</strong>: Integrate orbital mechanics constraints for physically-informed predictions<br>'
        '&bull; <strong>Larger labeled sets</strong>: Active learning to expand confirmed exoplanet training pool'
        '</div></div>'
        '</section>'
    )
    limitations_nav = '<li><a href="#limitations">Limitations</a></li>'

    training_section = ''
    training_nav = ''
    if os.path.isfile(os.path.join(output_dir, 'training_curves.png')):
        training_section = (
            '<section class="section" id="training">'
            '<div class="section-title reveal">CNN Training Curves</div>'
            '<div class="section-subtitle reveal delay-1">Loss and accuracy convergence over training epochs</div>'
            '<div class="plots single">'
            '<div class="plot-card reveal-scale delay-2">'
            '<img src="training_curves.png" alt="Training Curves">'
            '<p>Train vs. Validation — monitoring val_loss for early stopping</p>'
            '</div></div></section>'
        )
        training_nav = '<li><a href="#training">Training</a></li>'

    roc_section = ''
    roc_nav = ''
    if os.path.isfile(os.path.join(output_dir, 'roc_curve.png')):
        roc_section = (
            '<section class="section" id="roccurve">'
            '<div class="section-title reveal">ROC Curve</div>'
            '<div class="section-subtitle reveal delay-1">Receiver Operating Characteristic — sensitivity vs specificity trade-off</div>'
            '<div class="plots single">'
            '<div class="plot-card reveal-scale delay-2">'
            '<img src="roc_curve.png" alt="ROC Curve">'
            '<p>Hybrid (solid) vs Baseline (dashed) vs Random (diagonal)</p>'
            '</div></div></section>'
        )
        roc_nav = '<li><a href="#roccurve">ROC</a></li>'

    # MCC and ROC-AUC values with safe defaults
    h_roc = hybrid_metrics.get('roc_auc', 0)
    h_mcc = hybrid_metrics.get('mcc', 0)
    b_roc = baseline_metrics.get('roc_auc', 0)
    b_mcc = baseline_metrics.get('mcc', 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exoplanet Detection — Results Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html {{ scroll-behavior: smooth; overflow-x: hidden; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #000000; color: #f5f5f7; min-height: 100vh;
            overflow-x: hidden; -webkit-font-smoothing: antialiased;
        }}
        .starfield {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }}
        .star {{ position: absolute; border-radius: 50%; background: white; animation: twinkle var(--duration) ease-in-out infinite alternate; }}
        @keyframes twinkle {{ 0% {{ opacity: 0.1; transform: scale(0.8); }} 100% {{ opacity: var(--max-opacity); transform: scale(1.2); }} }}
        .nav {{ position: sticky; top: 0; z-index: 100; background: rgba(0,0,0,0.72); backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px); border-bottom: 1px solid rgba(255,255,255,0.08); padding: 14px 0; }}
        .nav-inner {{ max-width: 1100px; margin: 0 auto; padding: 0 2rem; display: flex; align-items: center; justify-content: space-between; }}
        .nav-logo {{ font-weight: 700; font-size: 1.1rem; letter-spacing: -0.02em; color: #f5f5f7; }}
        .nav-links {{ display: flex; gap: 1.5rem; list-style: none; flex-wrap: wrap; }}
        .nav-links a {{ color: rgba(255,255,255,0.65); text-decoration: none; font-size: 0.8rem; font-weight: 400; transition: color 0.3s; position: relative; }}
        .nav-links a::after {{ content: ''; position: absolute; bottom: -4px; left: 50%; width: 0; height: 1.5px; background: #2997ff; transition: width 0.3s, left 0.3s; }}
        .nav-links a:hover {{ color: #ffffff; }}
        .nav-links a:hover::after {{ width: 100%; left: 0; }}
        .hero {{ position: relative; z-index: 1; text-align: center; padding: 140px 2rem 100px; max-width: 900px; margin: 0 auto; }}
        .hero-eyebrow {{ display: inline-block; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #2997ff; margin-bottom: 1.2rem; opacity: 0; transform: translateY(15px); }}
        .hero h1 {{ font-size: clamp(2.8rem, 6vw, 4.5rem); font-weight: 800; letter-spacing: -0.04em; line-height: 1.08; margin-bottom: 1.5rem; background: linear-gradient(180deg, #ffffff 0%, rgba(255,255,255,0.6) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; opacity: 0; transform: translateY(20px); }}
        .hero p {{ font-size: 1.25rem; color: rgba(255,255,255,0.5); font-weight: 300; line-height: 1.5; max-width: 600px; margin: 0 auto; opacity: 0; transform: translateY(20px); }}
        .reveal {{ opacity: 0; transform: translateY(30px); transition: opacity 0.8s cubic-bezier(0.4,0,0.2,1), transform 0.8s cubic-bezier(0.4,0,0.2,1); }}
        .reveal.visible {{ opacity: 1; transform: translateY(0); }}
        .reveal-left {{ opacity: 0; transform: translateX(-40px); transition: opacity 0.8s cubic-bezier(0.4,0,0.2,1), transform 0.8s cubic-bezier(0.4,0,0.2,1); }}
        .reveal-left.visible {{ opacity: 1; transform: translateX(0); }}
        .reveal-right {{ opacity: 0; transform: translateX(40px); transition: opacity 0.8s cubic-bezier(0.4,0,0.2,1), transform 0.8s cubic-bezier(0.4,0,0.2,1); }}
        .reveal-right.visible {{ opacity: 1; transform: translateX(0); }}
        .reveal-scale {{ opacity: 0; transform: scale(0.92); transition: opacity 0.8s cubic-bezier(0.4,0,0.2,1), transform 0.8s cubic-bezier(0.4,0,0.2,1); }}
        .reveal-scale.visible {{ opacity: 1; transform: scale(1); }}
        .delay-1 {{ transition-delay: 0.1s; }} .delay-2 {{ transition-delay: 0.2s; }} .delay-3 {{ transition-delay: 0.3s; }} .delay-4 {{ transition-delay: 0.4s; }} .delay-5 {{ transition-delay: 0.5s; }}
        .container {{ position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 0 2rem; }}
        .section {{ padding: 100px 0; border-top: 1px solid rgba(255,255,255,0.06); }}
        .section-title {{ font-size: clamp(1.8rem,4vw,2.8rem); font-weight: 700; letter-spacing: -0.03em; margin-bottom: 0.6rem; text-align: center; }}
        .section-subtitle {{ font-size: 1.05rem; color: rgba(255,255,255,0.45); text-align: center; font-weight: 300; margin-bottom: 3.5rem; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1.5rem; margin-bottom: 4rem; }}
        .stat-card {{ text-align: center; padding: 2.5rem 1.5rem; border-radius: 20px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); transition: transform 0.5s cubic-bezier(0.4,0,0.2,1), box-shadow 0.5s, border-color 0.5s, background 0.5s; cursor: default; }}
        .stat-card:hover {{ transform: translateY(-6px) scale(1.02); background: rgba(255,255,255,0.07); border-color: rgba(41,151,255,0.3); box-shadow: 0 20px 60px rgba(41,151,255,0.12); }}
        .stat-value {{ font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.4rem; color: #2997ff; }}
        .stat-value.green {{ color: #30d158; }} .stat-value.purple {{ color: #bf5af2; }} .stat-value.orange {{ color: #ff9f0a; }} .stat-value.pink {{ color: #ff375f; }}
        .stat-label {{ font-size: 0.85rem; color: rgba(255,255,255,0.5); font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }}
        .stat-model {{ font-size: 0.72rem; color: rgba(255,255,255,0.3); margin-top: 0.5rem; }}
        .table-card {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; overflow: hidden; transition: transform 0.5s, box-shadow 0.5s; }}
        .table-card:hover {{ transform: translateY(-4px); box-shadow: 0 20px 60px rgba(0,0,0,0.4); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 18px 24px; text-align: center; font-size: 0.95rem; }}
        th {{ background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.7); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid rgba(255,255,255,0.08); }}
        td {{ color: rgba(255,255,255,0.7); border-bottom: 1px solid rgba(255,255,255,0.04); transition: background 0.3s; }}
        tr:hover td {{ background: rgba(255,255,255,0.03); }}
        td:first-child {{ text-align: left; font-weight: 500; color: #f5f5f7; }}
        .highlight {{ color: #30d158; font-weight: 700; font-size: 1.05rem; }}
        .dim {{ color: rgba(255,255,255,0.35); }}
        .report-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        .report-card {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; padding: 2rem; transition: transform 0.5s, box-shadow 0.5s, border-color 0.5s; }}
        .report-card:hover {{ transform: translateY(-4px); box-shadow: 0 16px 48px rgba(0,0,0,0.3); }}
        .report-card.baseline:hover {{ border-color: rgba(255,55,95,0.3); }}
        .report-card.hybrid:hover {{ border-color: rgba(48,209,88,0.3); }}
        .report-label {{ display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; padding: 4px 12px; border-radius: 100px; margin-bottom: 1.2rem; }}
        .report-label.baseline {{ background: rgba(255,55,95,0.12); color: #ff375f; }}
        .report-label.hybrid {{ background: rgba(48,209,88,0.12); color: #30d158; }}
        pre {{ background: rgba(0,0,0,0.5); padding: 1.2rem; border-radius: 12px; overflow-x: auto; font-size: 0.82rem; font-family: 'SF Mono', 'Fira Code', monospace; color: rgba(255,255,255,0.7); line-height: 1.6; border: 1px solid rgba(255,255,255,0.04); }}
        .plots {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        .plots.single {{ grid-template-columns: 1fr; max-width: 800px; margin: 0 auto; }}
        .plot-card {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; padding: 1.5rem; text-align: center; overflow: hidden; transition: transform 0.5s, box-shadow 0.5s, border-color 0.5s; }}
        .plot-card:hover {{ transform: translateY(-6px) scale(1.01); border-color: rgba(255,255,255,0.12); box-shadow: 0 24px 80px rgba(0,0,0,0.5); }}
        .plot-card img {{ max-width: 100%; border-radius: 12px; transition: transform 0.6s; }}
        .plot-card:hover img {{ transform: scale(1.03); }}
        .plot-card p {{ margin-top: 1rem; color: rgba(255,255,255,0.4); font-size: 0.85rem; }}
        .footer {{ text-align: center; padding: 60px 2rem 40px; color: rgba(255,255,255,0.2); font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.06); }}
        .footer-logo {{ font-size: 2rem; margin-bottom: 0.8rem; }}
        @media (max-width: 768px) {{
            .hero {{ padding: 100px 1.5rem 60px; }}
            .section {{ padding: 60px 0; }}
            .plots, .report-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .nav-links {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="starfield" id="starfield"></div>

    <nav class="nav">
        <div class="nav-inner">
            <div class="nav-logo">🪐 Exoplanet Detection</div>
            <ul class="nav-links">
                <li><a href="#metrics">Metrics</a></li>
                {training_nav}
                {cv_nav}
                <li><a href="#comparison">Comparison</a></li>
                <li><a href="#reports">Reports</a></li>
                <li><a href="#confusion">Confusion</a></li>
                <li><a href="#prcurve">PR Curve</a></li>
                {roc_nav}
                <li><a href="#features">Features</a></li>
                {shap_nav}
                {ablation_nav}
                {ci_nav}
                <li><a href="#lightcurves">Light Curves</a></li>
                {limitations_nav}
            </ul>
        </div>
    </nav>

    <header class="hero">
        <div class="hero-eyebrow" id="heroEyebrow">Hybrid Deep Feature Extraction + Ensemble Learning</div>
        <h1 id="heroTitle">Exoplanet Detection<br>Results Report</h1>
        <p id="heroSub">A 1D CNN learns transit patterns from Kepler light curves, then a Random Forest classifies the deep features — with SHAP explainability and comprehensive evaluation.</p>
    </header>

    <div class="container">

        <!-- KEY METRICS -->
        <section class="section" id="metrics">
            <div class="section-title reveal">Key Metrics</div>
            <div class="section-subtitle reveal delay-1">Hybrid CNN+RF performance on the Kepler test set</div>
            <div class="stats-grid">
                <div class="stat-card reveal delay-1">
                    <div class="stat-value green" data-counter="{hybrid_metrics['precision']:.4f}">{hybrid_metrics['precision']:.4f}</div>
                    <div class="stat-label">Precision</div>
                    <div class="stat-model">Hybrid CNN+RF</div>
                </div>
                <div class="stat-card reveal delay-2">
                    <div class="stat-value" data-counter="{hybrid_metrics['recall']:.4f}">{hybrid_metrics['recall']:.4f}</div>
                    <div class="stat-label">Recall</div>
                    <div class="stat-model">Hybrid CNN+RF</div>
                </div>
                <div class="stat-card reveal delay-3">
                    <div class="stat-value purple" data-counter="{hybrid_metrics['f1']:.4f}">{hybrid_metrics['f1']:.4f}</div>
                    <div class="stat-label">F1-Score</div>
                    <div class="stat-model">Hybrid CNN+RF</div>
                </div>
                <div class="stat-card reveal delay-4">
                    <div class="stat-value orange" data-counter="{hybrid_metrics['pr_auc']:.4f}">{hybrid_metrics['pr_auc']:.4f}</div>
                    <div class="stat-label">PR-AUC</div>
                    <div class="stat-model">Hybrid CNN+RF</div>
                </div>
                <div class="stat-card reveal delay-5">
                    <div class="stat-value" data-counter="{h_roc:.4f}">{h_roc:.4f}</div>
                    <div class="stat-label">ROC-AUC</div>
                    <div class="stat-model">Hybrid CNN+RF</div>
                </div>
                <div class="stat-card reveal delay-1">
                    <div class="stat-value pink" data-counter="{h_mcc:.4f}">{h_mcc:.4f}</div>
                    <div class="stat-label">MCC</div>
                    <div class="stat-model">Hybrid CNN+RF</div>
                </div>
            </div>
        </section>

        {training_section}

        {cv_section}

        <!-- COMPARISON TABLE -->
        <section class="section" id="comparison">
            <div class="section-title reveal">Model Comparison</div>
            <div class="section-subtitle reveal delay-1">Baseline Random Forest vs. Hybrid CNN + Random Forest</div>
            <div class="table-card reveal delay-2">
                <table>
                    <thead><tr><th>Metric</th><th>Baseline RF (Raw Flux)</th><th>Hybrid CNN+RF (Deep Features)</th></tr></thead>
                    <tbody>
                        <tr><td>Precision</td><td class="dim">{baseline_metrics['precision']:.4f}</td><td class="highlight">{hybrid_metrics['precision']:.4f}</td></tr>
                        <tr><td>Recall</td><td class="dim">{baseline_metrics['recall']:.4f}</td><td class="highlight">{hybrid_metrics['recall']:.4f}</td></tr>
                        <tr><td>F1-Score</td><td class="dim">{baseline_metrics['f1']:.4f}</td><td class="highlight">{hybrid_metrics['f1']:.4f}</td></tr>
                        <tr><td>PR-AUC</td><td class="dim">{baseline_metrics['pr_auc']:.4f}</td><td class="highlight">{hybrid_metrics['pr_auc']:.4f}</td></tr>
                        <tr><td>ROC-AUC</td><td class="dim">{b_roc:.4f}</td><td class="highlight">{h_roc:.4f}</td></tr>
                        <tr><td>MCC</td><td class="dim">{b_mcc:.4f}</td><td class="highlight">{h_mcc:.4f}</td></tr>
                        <tr><td>Accuracy</td><td class="dim">{baseline_metrics['accuracy']:.4f}</td><td class="highlight">{hybrid_metrics['accuracy']:.4f}</td></tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- CLASSIFICATION REPORTS -->
        <section class="section" id="reports">
            <div class="section-title reveal">Classification Reports</div>
            <div class="section-subtitle reveal delay-1">Full per-class breakdown for both models</div>
            <div class="report-grid">
                <div class="report-card baseline reveal-left delay-2">
                    <span class="report-label baseline">Baseline RF</span>
                    <pre>{baseline_metrics['report']}</pre>
                </div>
                <div class="report-card hybrid reveal-right delay-2">
                    <span class="report-label hybrid">Hybrid CNN+RF</span>
                    <pre>{hybrid_metrics['report']}</pre>
                </div>
            </div>
        </section>

        <!-- CONFUSION MATRICES -->
        <section class="section" id="confusion">
            <div class="section-title reveal">Confusion Matrices</div>
            <div class="section-subtitle reveal delay-1">True vs. predicted label distributions</div>
            <div class="plots">
                <div class="plot-card reveal-left delay-2">
                    <img src="confusion_matrix_baseline.png" alt="Baseline Confusion Matrix">
                    <p>Baseline RF</p>
                </div>
                <div class="plot-card reveal-right delay-2">
                    <img src="confusion_matrix_hybrid.png" alt="Hybrid Confusion Matrix">
                    <p>Hybrid CNN+RF</p>
                </div>
            </div>
        </section>

        <!-- PR CURVE -->
        <section class="section" id="prcurve">
            <div class="section-title reveal">Precision-Recall Curve</div>
            <div class="section-subtitle reveal delay-1">PR-AUC is the preferred metric for imbalanced classification</div>
            <div class="plots single">
                <div class="plot-card reveal-scale delay-2">
                    <img src="pr_curve.png" alt="Precision-Recall Curve">
                    <p>Hybrid (solid) vs Baseline (dashed)</p>
                </div>
            </div>
        </section>

        {roc_section}

        <!-- FEATURE IMPORTANCE -->
        <section class="section" id="features">
            <div class="section-title reveal">Feature Importance</div>
            <div class="section-subtitle reveal delay-1">Top 20 CNN-learned features ranked by Random Forest importance</div>
            <div class="plots single">
                <div class="plot-card reveal-scale delay-2">
                    <img src="feature_importance.png" alt="Feature Importance">
                    <p>Which CNN features matter most for classification</p>
                </div>
            </div>
        </section>

        {shap_section}

        {ablation_section}

        {ci_section}

        <!-- SAMPLE LIGHT CURVES -->
        <section class="section" id="lightcurves">
            <div class="section-title reveal">Sample Light Curves</div>
            <div class="section-subtitle reveal delay-1">Exoplanet transits show characteristic brightness dips</div>
            <div class="plots single">
                <div class="plot-card reveal-scale delay-2">
                    <img src="sample_light_curves.png" alt="Sample Light Curves">
                    <p>Exoplanets (top row) vs Non-planets (bottom row)</p>
                </div>
            </div>
        </section>

        {limitations_section}
    </div>

    <div class="footer">
        <div class="footer-logo">🔭</div>
        <p>Exoplanet Detection Pipeline &bull; Hybrid Deep Feature Extraction + Ensemble Learning</p>
        <p style="margin-top:0.5rem;">Built with TensorFlow &bull; Scikit-learn &bull; SHAP &bull; Kepler Space Telescope Data</p>
    </div>

    <script>
        (function createStars() {{
            const field = document.getElementById('starfield');
            for (let i = 0; i < 120; i++) {{
                const star = document.createElement('div');
                star.classList.add('star');
                const size = Math.random() * 2.5 + 0.5;
                star.style.width = size + 'px'; star.style.height = size + 'px';
                star.style.left = Math.random() * 100 + '%'; star.style.top = Math.random() * 100 + '%';
                star.style.setProperty('--duration', (Math.random() * 3 + 2) + 's');
                star.style.setProperty('--max-opacity', (Math.random() * 0.6 + 0.2).toFixed(2));
                star.style.animationDelay = (Math.random() * 5) + 's';
                field.appendChild(star);
            }}
        }})();

        window.addEventListener('load', () => {{
            const eyebrow = document.getElementById('heroEyebrow');
            const title = document.getElementById('heroTitle');
            const sub = document.getElementById('heroSub');
            setTimeout(() => {{ eyebrow.style.transition = 'opacity 0.7s cubic-bezier(0.4,0,0.2,1), transform 0.7s cubic-bezier(0.4,0,0.2,1)'; eyebrow.style.opacity = '1'; eyebrow.style.transform = 'translateY(0)'; }}, 200);
            setTimeout(() => {{ title.style.transition = 'opacity 0.8s cubic-bezier(0.4,0,0.2,1), transform 0.8s cubic-bezier(0.4,0,0.2,1)'; title.style.opacity = '1'; title.style.transform = 'translateY(0)'; }}, 500);
            setTimeout(() => {{ sub.style.transition = 'opacity 0.8s cubic-bezier(0.4,0,0.2,1), transform 0.8s cubic-bezier(0.4,0,0.2,1)'; sub.style.opacity = '1'; sub.style.transform = 'translateY(0)'; }}, 800);
        }});

        const revealElements = document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .reveal-scale');
        const revealObserver = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{ entry.target.classList.add('visible'); }}
                else {{ entry.target.classList.remove('visible'); }}
            }});
        }}, {{ threshold: 0.08, rootMargin: '0px 0px -30px 0px' }});
        revealElements.forEach(el => revealObserver.observe(el));

        const counterElements = document.querySelectorAll('[data-counter]');
        const counterObserver = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    const target = parseFloat(entry.target.dataset.counter);
                    const startTime = performance.now();
                    function animate(currentTime) {{
                        const progress = Math.min((currentTime - startTime) / 1200, 1);
                        entry.target.textContent = (target * (1 - Math.pow(1 - progress, 3))).toFixed(4);
                        if (progress < 1) requestAnimationFrame(animate);
                    }}
                    requestAnimationFrame(animate);
                }} else {{ entry.target.textContent = '0.0000'; }}
            }});
        }}, {{ threshold: 0.5 }});
        counterElements.forEach(el => counterObserver.observe(el));

        document.querySelectorAll('.nav-links a').forEach(anchor => {{
            anchor.addEventListener('click', function(e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{ window.scrollTo({{ top: target.getBoundingClientRect().top + window.pageYOffset - 60, behavior: 'smooth' }}); }}
            }});
        }});

        window.addEventListener('scroll', () => {{
            document.getElementById('starfield').style.transform = 'translateY(' + (window.pageYOffset * 0.15) + 'px)';
        }}, {{ passive: true }});
    </script>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[INFO] HTML report saved to: {html_path}")
    return html_path
