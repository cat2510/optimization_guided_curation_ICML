import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Iterable, Dict

def visualize_leaf_cost_distributions_4class(
        df_with_leaves: pd.DataFrame,
        leaf_col: str = 'leaf_assignment',
        cost_stratum_col: str = 'cost_stratum_2018',
        predicted_class_col: Optional[str] = 'predicted_cost_stratum',  # optional (same coding 0..3)
        cost_col: str = 'annual_cost_2018_deflated',
        strata: Iterable[int] = (0, 1, 2, 3),
        strata_labels: Optional[Dict[int, str]] = None,
        strata_colors: Optional[Dict[int, str]] = None,
        max_leaves_to_show: int = 40,
        select_method: str = 'largest',   # 'largest' | 'diverse' | 'auto'
        min_leaf_size: int = 1,
        annotate_counts: bool = True,
        show_percent: bool = True,
        figsize: Tuple[int, int] = (18, 10),
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
    """
    Visualize distribution of a 4-class cost stratum across leaves (OCT leaves).

    Parameters
    ----------
    df_with_leaves : DataFrame
        Data containing at least leaf_col, cost_stratum_col (values in strata), and optionally cost_col & predicted_class_col.
    leaf_col : str
        Column identifying leaf membership.
    cost_stratum_col : str
        Column with integer strata (e.g., 0,1,2,3).
    predicted_class_col : str or None
        Optional predicted strata column (same encoding). If None or absent, prediction summaries are skipped.
    strata : iterable
        All possible strata values (order = stacking order in the bar plot).
    strata_labels : dict
        Mapping stratum -> readable label. If None, labels = string of the integer.
    strata_colors : dict
        Mapping stratum -> color. If None, a seaborn palette is assigned.
    max_leaves_to_show : int
        Maximum number of leaves (bars) to display.
    select_method : str
        How to choose leaves:
          - 'largest': top N by sample size.
          - 'diverse': prioritize leaves with higher entropy (class diversity).
          - 'auto': combine largest + most pure for each stratum.
    min_leaf_size : int
        Filter out leaves with fewer than this many samples before selection.
    annotate_counts : bool
        Annotate each stacked segment with its count (and percent if show_percent).
    show_percent : bool
        Include percentage in segment annotations.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If provided, saves the figure (dpi=300, tight layout).
    verbose : bool
        Print textual summaries.

    Returns
    -------
    leaf_stats_df : DataFrame
        Per-leaf statistics (counts, rates, dominant stratum, cost summaries, optional prediction distribution).
    """

    df = df_with_leaves.copy()

    # Basic validation
    if cost_stratum_col not in df.columns:
        raise ValueError(f"{cost_stratum_col} not in DataFrame.")
    if leaf_col not in df.columns:
        raise ValueError(f"{leaf_col} not in DataFrame.")

    # Ensure stratum is integer/categorical
    df[cost_stratum_col] = df[cost_stratum_col].astype(int)

    # Default labels/colors
    if strata_labels is None:
        strata_labels = {s: str(s) for s in strata}
    if strata_colors is None:
        palette = sns.color_palette("Set2", n_colors=len(strata))
        strata_colors = {s: palette[i] for i, s in enumerate(strata)}

    # Collect per-leaf stats
    leaf_stats = []
    for leaf_id, leaf_data in df.groupby(leaf_col):
        n = len(leaf_data)
        if n < min_leaf_size:
            continue

        vc = leaf_data[cost_stratum_col].value_counts()
        stats = {
            'leaf_id': leaf_id,
            'total_samples': n
        }

        # Counts & rates per stratum
        for s in strata:
            cnt = int(vc.get(s, 0))
            stats[f'{cost_stratum_col}_{s}_count'] = cnt
            stats[f'{cost_stratum_col}_{s}_rate'] = cnt / n

        # Dominant stratum
        dominant = max(strata, key=lambda s: stats[f'{cost_stratum_col}_{s}_count'])
        stats['dominant_stratum'] = dominant

        # Entropy (diversity)
        probs = np.array([stats[f'{cost_stratum_col}_{s}_rate'] for s in strata if stats[f'{cost_stratum_col}_{s}_rate'] > 0])
        entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0
        stats['class_entropy'] = entropy

        # Cost summaries if available
        if cost_col in leaf_data.columns:
            stats['mean_cost'] = leaf_data[cost_col].mean()
            stats['median_cost'] = leaf_data[cost_col].median()
            stats['std_cost'] = leaf_data[cost_col].std()

        # Predicted distribution (optional)
        if predicted_class_col and predicted_class_col in leaf_data.columns:
            pvc = leaf_data[predicted_class_col].value_counts()
            for s in strata:
                stats[f'pred_{cost_stratum_col}_{s}_count'] = int(pvc.get(s, 0))
                stats[f'pred_{cost_stratum_col}_{s}_rate'] = stats[f'pred_{cost_stratum_col}_{s}_count'] / n
            pred_dom = max(strata, key=lambda s: stats.get(f'pred_{cost_stratum_col}_{s}_count', 0))
            stats['dominant_pred_stratum'] = pred_dom

        leaf_stats.append(stats)

    if not leaf_stats:
        raise ValueError("No leaves passed the filtering criteria (check min_leaf_size or column names).")

    leaf_stats_df = pd.DataFrame(leaf_stats).sort_values('total_samples', ascending=False)

    # Leaf selection strategies
    if select_method == 'largest':
        selected = leaf_stats_df.head(max_leaves_to_show)
    elif select_method == 'diverse':
        selected = leaf_stats_df.sort_values('class_entropy', ascending=False).head(max_leaves_to_show)
    elif select_method == 'auto':
        # Combine largest + most pure for each stratum
        largest = leaf_stats_df.head(max_leaves_to_show // 2)
        purest_list = []
        for s in strata:
            col_rate = f'{cost_stratum_col}_{s}_rate'
            purest_s = (leaf_stats_df.sort_values(col_rate, ascending=False)
                        .head(max(1, max_leaves_to_show // (2 * len(strata)))))
            purest_list.append(purest_s)
        combined = pd.concat([largest] + purest_list, ignore_index=True)
        selected = combined.drop_duplicates('leaf_id').head(max_leaves_to_show)
    else:
        raise ValueError("select_method must be one of: 'largest', 'diverse', 'auto'.")

    # Sort selected leaves by total_samples (can customize)
    selected = selected.sort_values('total_samples', ascending=False).reset_index(drop=True)

    # ------------------ Plot: Single stacked bar distribution ------------------
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(len(selected))
    bottoms = np.zeros(len(selected))

    # For consistent stacking order use strata as given
    for s in strata:
        count_col = f'{cost_stratum_col}_{s}_count'
        if count_col not in selected.columns:
            heights = np.zeros(len(selected))
        else:
            heights = selected[count_col].values

        bars = ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=strata_colors[s],
            edgecolor='black',
            linewidth=0.4,
            label=strata_labels.get(s, str(s)),
            alpha=0.85
        )

        if annotate_counts:
            for xi, h, b in zip(x, heights, bottoms):
                if h <= 0:
                    continue
                pct = h / selected.loc[xi, 'total_samples']
                label = f"{int(h)}"
                if show_percent:
                    label += f"\n({pct*100:.1f}%)"
                ax.text(
                    xi,
                    b + h / 2,
                    label,
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black'
                )

        bottoms += heights

    # X tick labels: Leaf ID (n)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"L{row.leaf_id}\n(n={row.total_samples})" for _, row in selected.iterrows()],
        rotation=45, ha='right', fontsize=10
    )

    ax.set_ylabel("Number of Samples", fontsize=14)
    ax.set_xlabel("Leaves", fontsize=14)
    ax.set_title("Cost Stratum Distribution per Leaf (Stacked Counts)", fontsize=16)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.legend(title="Cost Stratum", fontsize=10, title_fontsize=11,
              bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"Plot saved to: {save_path}")
    plt.show()

    # ------------------ Summary (optional) ------------------
    if verbose:
        print("\n" + "=" * 80)
        print("LEAF STRATUM SUMMARY")
        print("=" * 80)
        # Overall distribution
        overall_counts = {f'{cost_stratum_col}_{s}_count': leaf_stats_df[f'{cost_stratum_col}_{s}_count'].sum()
                          for s in strata}
        overall_total = sum(overall_counts.values())
        print("Overall class distribution across all leaves:")
        for s in strata:
            c = overall_counts[f'{cost_stratum_col}_{s}_count']
            print(f"  Stratum {s} ({strata_labels.get(s,str(s))}): {c} ({c/overall_total*100:.1f}%)")

        print("\nDominant stratum frequency among selected leaves:")
        dom_counts = selected['dominant_stratum'].value_counts()
        for s, c in dom_counts.items():
            print(f"  Stratum {s} ({strata_labels.get(s,str(s))}): {c} leaves ({c/len(selected)*100:.1f}%)")

        if predicted_class_col and f'pred_{cost_stratum_col}_{strata[0]}_count' in leaf_stats_df.columns:
            print("\nMean predicted vs actual rates (overall):")
            rows = []
            for s in strata:
                actual_rate = (leaf_stats_df[f'{cost_stratum_col}_{s}_count'].sum() /
                               leaf_stats_df['total_samples'].sum())
                pred_rate = (leaf_stats_df[f'pred_{cost_stratum_col}_{s}_count'].sum() /
                             leaf_stats_df['total_samples'].sum())
                rows.append((s, strata_labels.get(s,str(s)), actual_rate, pred_rate))
            df_rates = pd.DataFrame(rows, columns=['Stratum', 'Label', 'Actual_Rate', 'Pred_Rate'])
            print(df_rates.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return leaf_stats_df