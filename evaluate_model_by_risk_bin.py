import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve
import re


def plot_probability_and_calibration(
    clf, X_test, y_test, risk_bin_column, highcost_cutoff=4, figsize=(10, 4)):
    """
    plots the probability distribution and calibration plot.
    """
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_highcost = np.asarray(y_test)

    df = pd.DataFrame({
        'high_cost': y_highcost,
        'probability': y_proba,
        'risk_bin': risk_bin_column.values
    })

    def extract_lower_bound(bin_str):
        match = re.search(r'(\d+\.\d+)-', bin_str)
        if match:
            return float(match.group(1))
        else:
            return 0.0

    unique_bins = sorted(df['risk_bin'].unique(), key=extract_lower_bound)
    bin_order = {bin_name: i for i, bin_name in enumerate(unique_bins)}
    df['bin_order'] = df['risk_bin'].map(bin_order)

    # Aggregate for calibration plot
    bin_df = []
    for bin_name in unique_bins:
        bin_data = df[df['risk_bin'] == bin_name]
        if len(bin_data) == 0:
            continue
        try:
            lower, upper = map(float, bin_name.split('-'))
            bin_range = (lower, upper)
        except:
            bin_range = (0, 0)
        bin_df.append({
            'bin': bin_name,
            'range': bin_range,
            'count': len(bin_data),
            'avg_probability': bin_data['probability'].mean(),
            'high_cost_pct': bin_data['high_cost'].mean() * 100
        })
    bin_df = pd.DataFrame(bin_df)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Probability distribution
    sns.histplot(
        x='probability', hue='high_cost', data=df,
        element='step', stat='density', common_norm=False,
        ax=axes[0]
    )
    axes[0].set_title('Predicted Probability Distribution by True Status')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Density')

    # Calibration plot
    if len(bin_df) > 0:
        axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        axes[1].scatter(bin_df['avg_probability'], bin_df['high_cost_pct']/100, 
                        s=bin_df['count']/len(df)*1000, alpha=0.6)
        for i, row in bin_df.iterrows():
            axes[1].annotate(row['bin'], 
                             (row['avg_probability'], row['high_cost_pct']/100),
                             xytext=(5, 5), textcoords='offset points')
        axes[1].set_title('Calibration Plot (Bin Size ~ Circle Size)')
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Observed High-Cost Rate')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def evaluate_metrics_by_risk_bin(clf, X_test, y_test, risk_bin_column, optimal_threshold=True, model_name=None, dataset_name=None):
    """
    Evaluates recall, specificity, precision, f1, and AUC across risk bins.
    Returns a DataFrame with all metrics, including model_name/dataset_name if provided.
    """
    if model_name == "oct":
        print("OCT evaluation")
        y_proba = clf.predict_proba(X_test)["1"]

    else:
        y_proba = clf.predict_proba(X_test)[:, 1]
    y_highcost = np.asarray(y_test)

    df = pd.DataFrame({
        'high_cost': y_highcost,
        'probability': y_proba,
        'risk_bin': risk_bin_column.values
    })
    # Filter out NaN risk bins
    df = df.dropna(subset=['risk_bin'])
    
    # Get unique bins (no sorting needed)
    unique_bins = df['risk_bin'].unique()
    results = []
    ## Youden thresholding 
    if optimal_threshold:
        precision, recall, thresholds = precision_recall_curve(y_highcost, y_proba)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        threshold = best_threshold
    else:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_highcost, y_pred, labels=[0, 1]).ravel()

    overall_auc = roc_auc_score(y_highcost, y_proba)
    overall_results = {
        'bin': 'Overall',
        'count': len(df),
        'high_cost_count': np.sum(y_highcost),
        'high_cost_pct': np.mean(y_highcost) * 100,
        'avg_probability': np.mean(y_proba),
        'threshold': threshold,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1': 2 * (tp / (tp + fp) * tp / (tp + fn)) / (tp / (tp + fp) + tp / (tp + fn)) if (tp + fp) > 0 and (tp + fn) > 0 else 0,
        'auc': overall_auc
    }
    results.append(overall_results)

    for bin_name in unique_bins:
        bin_df = df[df['risk_bin'] == bin_name]
        if len(bin_df) == 0:
            continue
        bin_results = {
            'bin': bin_name,
            'count': len(bin_df),
            'count_pct': len(bin_df) / len(df) * 100,
            'high_cost_count': bin_df['high_cost'].sum(),
            'high_cost_pct': bin_df['high_cost'].mean() * 100,
            'avg_probability': bin_df['probability'].mean(),
        }
        bin_y_pred = (bin_df['probability'] >= threshold).astype(int)
        bin_y_true = bin_df['high_cost']
        if len(np.unique(bin_y_true)) == 1:
            # Skip this bin or handle specially
            print(f"Risk bin {bin_name} contains only one class, skipping confusion matrix")
            continue
        if len(bin_y_true) > 0:
            if bin_y_true.sum() > 0:
                bin_results['recall'] = recall_score(bin_y_true, bin_y_pred, zero_division=0)
            else:
                bin_results['recall'] = np.nan
            if (len(bin_y_true) - bin_y_true.sum()) > 0:
                tn, fp, fn, tp = confusion_matrix(bin_y_true, bin_y_pred).ravel()
                bin_results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                bin_results['specificity'] = np.nan
            bin_results['precision'] = precision_score(bin_y_true, bin_y_pred, zero_division=0)
            bin_results['f1'] = f1_score(bin_y_true, bin_y_pred, zero_division=0)
            # Bin AUC
            n_pos = bin_y_true.sum()
            n_neg = len(bin_y_true) - n_pos
            if n_pos > 0 and n_neg > 0:
                bin_results['auc'] = roc_auc_score(bin_y_true, bin_df['probability'])
            else:
                bin_results['auc'] = np.nan
        results.append(bin_results)

    results_df = pd.DataFrame(results)
    if model_name is not None:
        results_df['model_name'] = model_name
    if dataset_name is not None:
        results_df['dataset_name'] = dataset_name
    return results_df

def plot_model_comparison_by_risk_bin(results_df, metric='recall', figsize=(12, 5)):
    """
    Plots a line plot (with dots) of the given metric by risk bin
    for each model and dataset combination, with weighted average in the legend.
    """
    # Only use bin-specific rows (not 'Overall')
    plot_df = results_df[results_df['bin'] != 'Overall'].copy()

    # Create a unique group label for each line
    plot_df['label'] = plot_df['model_name'] + ' (' + plot_df['dataset_name'] + ')'

    # Calculate weighted average for each group/label
    weighted_avg_scores = {}
    for label, group in plot_df.groupby('label'):
        if group['high_cost_count'].sum() > 0:
            weighted_avg = (group[metric].fillna(0) * group['high_cost_count']).sum() / group['high_cost_count'].sum()
        else:
            weighted_avg = float('nan')
        weighted_avg_scores[label] = weighted_avg

    plt.figure(figsize=figsize)
    lines = []
    labels = []
    for label, group in plot_df.groupby('label'):
        group_sorted = group.sort_values('lower_bound')
        line, = plt.plot(
            group_sorted['bin'],
            group_sorted[metric],
            marker='o',
            label=None  # We'll manually set legend
        )
        # Create legend label with weighted average
        avg = weighted_avg_scores[label]
        legend_label = f"{label} (weighted avg={avg:.2f})"
        lines.append(line)
        labels.append(legend_label)

    plt.xlabel('Risk Bin')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} by Risk Bin')
    plt.ylim(0, 1)
    plt.legend(lines, labels, title='Model (Dataset)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()