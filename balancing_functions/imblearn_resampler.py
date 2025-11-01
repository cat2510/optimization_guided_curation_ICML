import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import Counter
from sklearn.impute import SimpleImputer
from imblearn.base import SamplerMixin


class RiskBinnedResampler:
    def __init__(
        self,
        bin_edges: Optional[np.ndarray] = None,
        impute_strategy: str = "median",
        uid_col: str = "uid",
        sampler: Optional[SamplerMixin] = None,
    ):
        self.bin_edges = (
            bin_edges if bin_edges is not None else np.linspace(0.0, 1.0, 11)
        )
        self.bin_labels = [
            f"{a:.2f}-{b:.2f}" for a, b in zip(self.bin_edges[:-1], self.bin_edges[1:])
        ]
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.uid_col = uid_col
        self.sampler = sampler

    def assign_bins(self, df: pd.DataFrame, proba_col: str = "prob") -> pd.DataFrame:
        df = df.copy()
        df["risk_bin"] = pd.cut(
            df[proba_col],
            bins=self.bin_edges,
            labels=self.bin_labels,
            include_lowest=True,
            right=False,
        )
        return df

    def impute_features(
        self, df: pd.DataFrame, exclude_cols: List[str]
    ) -> pd.DataFrame:
        """
        Impute *only* the feature columns, leaving exclude_cols (like 'uid', 'true_class', 'risk_bin',"prob")
        unchanged, then append them back.
        """
        df = df.copy()
        X = df.drop(columns=exclude_cols)
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X), columns=X.columns, index=X.index
        )
        # re-attach the excluded columns
        for col in exclude_cols:
            X_imputed[col] = df[col]
        return X_imputed

    def apply_sampler_by_bin(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, List]]:
        """
        Returns:
          - A resampled DataFrame (with 'uid' preserved)
          - A dict mapping each bin label --> list of uids that were removed
        """
        removed_ids: Dict[str, List] = {}
        resampled_parts = []

        for b in df["risk_bin"].dropna().unique():
            bin_data = df[df["risk_bin"] == b].copy()
            # 1) Impute
            bin_imp = self.impute_features(
                bin_data, exclude_cols=[self.uid_col, "true_class", "risk_bin", "prob"]
            )

            # 2) Split off uid / labels
            uids = bin_imp[self.uid_col].values
            X = bin_imp.drop(columns=[self.uid_col, "true_class", "risk_bin", "prob"])
            y = bin_imp["true_class"].values

            original_uids = set(uids)
            if len(set(y)) < 2:
                # nothing to resample
                removed_ids[b] = []
                continue

            # 3) Fit/resample
            X_res, y_res = self.sampler.fit_resample(X, y)

            # 4) Figure out UID assignment
            #    — if the sampler preserved DataFrame structure and had a 'uid' column, use it
            #    — else, for pure undersamplers, try to pull sample_indices_ attribute
            #    — otherwise assume no removals (SMOTE-style)
            if isinstance(X_res, pd.DataFrame) and self.uid_col in X_res.columns:
                uids_res = X_res[self.uid_col].values
                X_res = X_res.drop(columns=[self.uid_col])
            elif hasattr(self.sampler, "sample_indices_"):
                kept_idx = np.array(self.sampler.sample_indices_, dtype=int)
                uids_res = uids[kept_idx]
            else:
                # we can’t tell—assume oversampling only
                uids_res = uids

            # 5) Track removed
            removed_ids[b] = list(original_uids - set(uids_res))

            # 6) Assemble DataFrame
            df_part = pd.DataFrame(X_res, columns=X.columns)
            df_part["true_class"] = y_res
            df_part["risk_bin"] = b
            df_part[self.uid_col] = uids_res
            resampled_parts.append(df_part)

        final_df = (
            pd.concat(resampled_parts)
            .reset_index(drop=True)
            .astype({self.uid_col: df[self.uid_col].dtype})
        )
        return final_df, removed_ids

    def apply_sampler_global(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """
        Same idea applied to the entire dataset at once.
        Returns:
          - Resampled DataFrame (with 'uid')
          - List of uids that were removed
        """
        # 1) Impute everything
        df_imp = self.impute_features(
            df, exclude_cols=[self.uid_col, "true_class", "risk_bin", "prob"]
        )

        uids = df_imp[self.uid_col].values
        X = df_imp.drop(columns=[self.uid_col, "true_class", "risk_bin", "prob"])
        y = df_imp["true_class"].values

        original_uids = set(uids)
        if len(set(y)) < 2:
            return df.copy(), []

        # 2) Resample
        X_res, y_res = self.sampler.fit_resample(X, y)

        # 3) UID logic (as above)
        if isinstance(X_res, pd.DataFrame) and self.uid_col in X_res.columns:
            uids_res = X_res[self.uid_col].values
            X_res = X_res.drop(columns=[self.uid_col])
        elif hasattr(self.sampler, "sample_indices_"):
            kept_idx = np.array(self.sampler.sample_indices_, dtype=int)
            uids_res = uids[kept_idx]
        else:
            uids_res = uids

        removed = list(original_uids - set(uids_res))

        # 4) Assemble
        df_part = pd.DataFrame(X_res, columns=X.columns)
        df_part["true_class"] = y_res
        df_part["risk_bin"] = (
            pd.cut(
                df_part["prob"],
                bins=self.bin_edges,
                labels=self.bin_labels,
                include_lowest=True,
                right=False,
            )
            if "prob" in df_part.columns
            else np.nan
        )
        df_part[self.uid_col] = uids_res

        return df_part.reset_index(drop=True), removed
