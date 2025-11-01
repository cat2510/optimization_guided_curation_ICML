import pandas as pd
from typing import Union
from imblearn.under_sampling import (
    RandomUnderSampler,
    EditedNearestNeighbours,
    NeighbourhoodCleaningRule,
)


def reduce_dominant_bins(
    df: pd.DataFrame,
    bin_labels: Union[str, list] = "0.00-0.10",
    strategy: str = "random",
    target_ratio: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Undersample one or multiple risk bins to reduce majority class samples.

    For each specified bin, majority class will be reduced to:
        n_majority_after = target_ratio * n_minority

    Parameters:
    - df: full DataFrame with columns 'risk_bin', 'true_class', 'uid', 'prob'
    - bin_labels: string or list of bin labels to trim (e.g. "0.00-0.10" or ["0.00-0.10", "0.10-0.20"])
    - strategy: 'random', 'enn', or 'ncr'
        - 'random' uses RandomUnderSampler with target_ratio rule
        - 'enn' uses EditedNearestNeighbours
        - 'ncr' uses NeighbourhoodCleaningRule
    - target_ratio: float, ratio of final majority:minority samples in bin (only for 'random')
    - random_state: for reproducibility (only for 'random')

    Returns:
    - DataFrame with trimmed bins + untouched rest of data
    """
    if isinstance(bin_labels, str):
        bin_labels = [bin_labels]

    df_trimmed_list = []
    df_rest = df[~df["risk_bin"].isin(bin_labels)].copy()

    for bin_label in bin_labels:
        df_to_trim = df[df["risk_bin"] == bin_label].copy()
        if df_to_trim.empty:
            print(f"[{bin_label}] Skipping: bin is empty")
            continue

        X_trim = df_to_trim.drop(columns=["true_class", "risk_bin", "uid", "prob"])
        y_trim = pd.Series(df_to_trim["true_class"])

        if strategy == "random":
            class_counts = y_trim.value_counts()
            if len(class_counts) < 2:
                print(f"[{bin_label}] Skipping: only one class present")
                df_trimmed_list.append(df_to_trim)
                continue

            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            n_minority = class_counts[minority_class]
            n_majority = class_counts[majority_class]
            desired_majority = int(target_ratio * n_minority)

            if n_majority <= desired_majority:
                print(f"[{bin_label}] Skipping: majority already below target")
                df_trimmed_list.append(df_to_trim)
                continue

            sampler = RandomUnderSampler(
                sampling_strategy='auto', random_state=random_state
            )

        elif strategy == "enn":
            sampler = EditedNearestNeighbours()
        elif strategy == "ncr":
            sampler = NeighbourhoodCleaningRule()
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        try:
            X_res, y_res, *_ = sampler.fit_resample(X_trim, y_trim)
        except Exception as e:
            print(f"[{bin_label}] Skipping due to sampling error: {e}")
            df_trimmed_list.append(df_to_trim)
            continue

        df_trimmed = pd.DataFrame(X_res, columns=X_trim.columns)
        df_trimmed["true_class"] = y_res
        df_trimmed["risk_bin"] = bin_label
        df_trimmed["uid"] = [f"{bin_label}_trimmed_{i}" for i in range(len(y_res))]
        df_trimmed["prob"] = pd.Series(df_to_trim["prob"]).iloc[: len(df_trimmed)].values

        df_trimmed_list.append(df_trimmed)

    df_reduced = pd.DataFrame(pd.concat([df_rest] + df_trimmed_list, ignore_index=True))
    return df_reduced
