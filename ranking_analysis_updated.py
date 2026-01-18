# ============================================================================
# RANK CONFIGURATIONS BY AUC, PR-AUC, AND MCC
# ============================================================================

import pandas as pd
import numpy as np

# Load results
metrics_path = "kcenter_hyperparameter_search_results_global/metrics_master.csv"
table_ranking = pd.read_csv(metrics_path)

print("="*80)
print("MODEL RANKING BY AUC, PR-AUC, AND MCC")
print("="*80)
print(f"\nTotal configurations: {len(table_ranking)}")
print(f"\nAvailable metrics:")
available_metrics = ['auc', 'pr_auc', 'best_mcc', 'balanced_recall_gmean', 
                     'balanced_specificity_gmean', 'optimal_f1']
for metric in available_metrics:
    if metric in table_ranking.columns:
        print(f"  ✓ {metric}")
    else:
        print(f"  ✗ {metric} (not available)")

# Helper function to format display
def format_display_table(df, display_cols):
    """Format a dataframe for display with proper formatting."""
    df_formatted = df[display_cols].copy()
    
    # Format numeric columns
    numeric_cols = ['auc', 'pr_auc', 'best_mcc', 'optimal_f1', 
                    'balanced_recall_gmean', 'balanced_specificity_gmean',
                    'recall_mcc', 'precision_mcc', 'precision_gmean']
    for col in numeric_cols:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
            )
    
    # Format boolean
    if 'use_adaptive_pool' in df_formatted.columns:
        df_formatted['use_adaptive_pool'] = df_formatted['use_adaptive_pool'].apply(
            lambda x: 'Adaptive' if x == True else 'Fixed' if x == False else x
        )
    
    # Format None
    if 'case_weighting' in df_formatted.columns:
        df_formatted['case_weighting'] = df_formatted['case_weighting'].fillna('None')
    
    return df_formatted

# ============================================================================
# 1. RANKING BY AUC
# ============================================================================
print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS BY AUC")
print("="*80)

if 'auc' in table_ranking.columns:
    sorted_by_auc = table_ranking.sort_values('auc', ascending=False)
    
    display_cols = ['config_name', 'case_weighting', 'use_adaptive_pool', 'seed_method',
                   'auc', 'pr_auc', 'best_mcc', 
                   'balanced_recall_gmean', 'balanced_specificity_gmean', 'optimal_f1']
    display_cols = [c for c in display_cols if c in sorted_by_auc.columns]
    
    top10_auc = format_display_table(sorted_by_auc.head(10), display_cols)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    print(top10_auc.to_string(index=False))
    
    best_by_auc = sorted_by_auc.iloc[0]
    print(f"\n🏆 BEST BY AUC: {best_by_auc['config_name']}")
    print(f"   AUC: {best_by_auc['auc']:.4f}")
    print(f"   PR-AUC: {best_by_auc.get('pr_auc', 'N/A'):.4f}" if pd.notna(best_by_auc.get('pr_auc')) else f"   PR-AUC: N/A")
    print(f"   MCC: {best_by_auc.get('best_mcc', 'N/A'):.4f}" if pd.notna(best_by_auc.get('best_mcc')) else f"   MCC: N/A")
else:
    print("⚠️ 'auc' column not found")

# ============================================================================
# 2. RANKING BY PR-AUC
# ============================================================================
print("\n\n" + "="*80)
print("TOP 10 CONFIGURATIONS BY PR-AUC")
print("="*80)

if 'pr_auc' in table_ranking.columns:
    sorted_by_pr_auc = table_ranking.sort_values('pr_auc', ascending=False)
    
    display_cols = ['config_name', 'case_weighting', 'use_adaptive_pool', 'seed_method',
                   'pr_auc', 'auc', 'best_mcc',
                   'balanced_recall_gmean', 'balanced_specificity_gmean', 'optimal_f1']
    display_cols = [c for c in display_cols if c in sorted_by_pr_auc.columns]
    
    top10_pr_auc = format_display_table(sorted_by_pr_auc.head(10), display_cols)
    
    print(top10_pr_auc.to_string(index=False))
    
    best_by_pr_auc = sorted_by_pr_auc.iloc[0]
    print(f"\n🏆 BEST BY PR-AUC: {best_by_pr_auc['config_name']}")
    print(f"   PR-AUC: {best_by_pr_auc['pr_auc']:.4f}")
    print(f"   AUC: {best_by_pr_auc.get('auc', 'N/A'):.4f}" if pd.notna(best_by_pr_auc.get('auc')) else f"   AUC: N/A")
    print(f"   MCC: {best_by_pr_auc.get('best_mcc', 'N/A'):.4f}" if pd.notna(best_by_pr_auc.get('best_mcc')) else f"   MCC: N/A")
    print(f"   Recall (G-mean): {best_by_pr_auc.get('balanced_recall_gmean', 'N/A'):.4f}" if pd.notna(best_by_pr_auc.get('balanced_recall_gmean')) else f"   Recall (G-mean): N/A")
    print(f"   Specificity (G-mean): {best_by_pr_auc.get('balanced_specificity_gmean', 'N/A'):.4f}" if pd.notna(best_by_pr_auc.get('balanced_specificity_gmean')) else f"   Specificity (G-mean): N/A")
else:
    print("⚠️ 'pr_auc' column not found")

# ============================================================================
# 3. RANKING BY MCC
# ============================================================================
print("\n\n" + "="*80)
print("TOP 10 CONFIGURATIONS BY MCC (MATTHEWS CORRELATION COEFFICIENT)")
print("="*80)

if 'best_mcc' in table_ranking.columns:
    sorted_by_mcc = table_ranking.sort_values('best_mcc', ascending=False)
    
    # Check if recall/precision at MCC are available
    has_recall_mcc = 'recall_mcc' in table_ranking.columns
    has_precision_mcc = 'precision_mcc' in table_ranking.columns
    
    display_cols = ['config_name', 'case_weighting', 'use_adaptive_pool', 'seed_method',
                   'best_mcc', 'auc', 'pr_auc', 'optimal_f1']
    
    # Add MCC-specific metrics if available
    if has_recall_mcc:
        display_cols.append('recall_mcc')
    if has_precision_mcc:
        display_cols.append('precision_mcc')
    
    # Add G-mean metrics
    display_cols.extend(['balanced_recall_gmean', 'balanced_specificity_gmean'])
    
    display_cols = [c for c in display_cols if c in sorted_by_mcc.columns]
    
    top10_mcc = format_display_table(sorted_by_mcc.head(10), display_cols)
    
    print(top10_mcc.to_string(index=False))
    
    best_by_mcc = sorted_by_mcc.iloc[0]
    print(f"\n🏆 BEST BY MCC: {best_by_mcc['config_name']}")
    print(f"   MCC: {best_by_mcc['best_mcc']:.4f}")
    print(f"   AUC: {best_by_mcc.get('auc', 'N/A'):.4f}" if pd.notna(best_by_mcc.get('auc')) else f"   AUC: N/A")
    print(f"   PR-AUC: {best_by_mcc.get('pr_auc', 'N/A'):.4f}" if pd.notna(best_by_mcc.get('pr_auc')) else f"   PR-AUC: N/A")
    
    if has_recall_mcc:
        print(f"   Recall @ MCC: {best_by_mcc.get('recall_mcc', 'N/A'):.4f}" if pd.notna(best_by_mcc.get('recall_mcc')) else f"   Recall @ MCC: N/A")
    else:
        print(f"   ⚠️ Recall @ MCC: Not available in metrics (needs to be added to evaluate_binary_oct)")
    
    if has_precision_mcc:
        print(f"   Precision @ MCC: {best_by_mcc.get('precision_mcc', 'N/A'):.4f}" if pd.notna(best_by_mcc.get('precision_mcc')) else f"   Precision @ MCC: N/A")
    else:
        print(f"   ⚠️ Precision @ MCC: Not available in metrics (needs to be added to evaluate_binary_oct)")
    
    print(f"   Recall (G-mean): {best_by_mcc.get('balanced_recall_gmean', 'N/A'):.4f}" if pd.notna(best_by_mcc.get('balanced_recall_gmean')) else f"   Recall (G-mean): N/A")
    print(f"   Specificity (G-mean): {best_by_mcc.get('balanced_specificity_gmean', 'N/A'):.4f}" if pd.notna(best_by_mcc.get('balanced_specificity_gmean')) else f"   Specificity (G-mean): N/A")
else:
    print("⚠️ 'best_mcc' column not found")

# ============================================================================
# 4. SUMMARY TABLE: TOP 5 BY EACH METRIC WITH RECALL/PRECISION AT MCC & GMEAN
# ============================================================================
print("\n\n" + "="*80)
print("SUMMARY: TOP 5 BY EACH METRIC WITH RECALL/PRECISION AT MCC & GMEAN")
print("="*80)

summary_tables = []

# Top 5 by AUC
if 'auc' in table_ranking.columns:
    top5_auc = table_ranking.sort_values('auc', ascending=False).head(5)
    summary_cols = ['config_name', 'auc', 'pr_auc', 'best_mcc']
    if 'recall_mcc' in table_ranking.columns:
        summary_cols.append('recall_mcc')
    if 'precision_mcc' in table_ranking.columns:
        summary_cols.append('precision_mcc')
    summary_cols.extend(['balanced_recall_gmean', 'balanced_specificity_gmean'])
    summary_cols = [c for c in summary_cols if c in top5_auc.columns]
    
    summary_tables.append(("Top 5 by AUC", top5_auc[summary_cols]))

# Top 5 by PR-AUC
if 'pr_auc' in table_ranking.columns:
    top5_pr_auc = table_ranking.sort_values('pr_auc', ascending=False).head(5)
    summary_cols = ['config_name', 'pr_auc', 'auc', 'best_mcc']
    if 'recall_mcc' in table_ranking.columns:
        summary_cols.append('recall_mcc')
    if 'precision_mcc' in table_ranking.columns:
        summary_cols.append('precision_mcc')
    summary_cols.extend(['balanced_recall_gmean', 'balanced_specificity_gmean'])
    summary_cols = [c for c in summary_cols if c in top5_pr_auc.columns]
    
    summary_tables.append(("Top 5 by PR-AUC", top5_pr_auc[summary_cols]))

# Top 5 by MCC
if 'best_mcc' in table_ranking.columns:
    top5_mcc = table_ranking.sort_values('best_mcc', ascending=False).head(5)
    summary_cols = ['config_name', 'best_mcc', 'auc', 'pr_auc']
    if 'recall_mcc' in table_ranking.columns:
        summary_cols.append('recall_mcc')
    if 'precision_mcc' in table_ranking.columns:
        summary_cols.append('precision_mcc')
    summary_cols.extend(['balanced_recall_gmean', 'balanced_specificity_gmean'])
    summary_cols = [c for c in summary_cols if c in top5_mcc.columns]
    
    summary_tables.append(("Top 5 by MCC", top5_mcc[summary_cols]))

# Display all summary tables
for title, df in summary_tables:
    print(f"\n{title}:")
    print("-" * 80)
    df_formatted = format_display_table(df, df.columns.tolist())
    print(df_formatted.to_string(index=False))

# Note about missing metrics
print("\n" + "="*80)
print("NOTE: RECALL/PRECISION AT MCC")
print("="*80)
if 'recall_mcc' not in table_ranking.columns or 'precision_mcc' not in table_ranking.columns:
    print("⚠️ Recall and Precision at MCC threshold are not currently saved in the metrics CSV.")
    print("   These metrics are computed in evaluate_binary_oct() but not returned in the metrics dict.")
    print("   To include them, update model_IAI.py to return 'recall_mcc' and 'precision_mcc' in the metrics dictionary.")
else:
    print("✓ Recall and Precision at MCC threshold are available in the metrics.")

print("\n" + "="*80)
print("✅ Ranking analysis complete!")
print("="*80)
