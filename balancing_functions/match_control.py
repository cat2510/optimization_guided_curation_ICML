import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Union, Any
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from traitlets import Bool

from model_pipeline import get_preprocessor,get_bin_flag_columns


class RiskBinnedCaseControlResampler:
    def __init__(
        self,
        bin_edges: Optional[np.ndarray] = None,
        impute_strategy: str = "median",
        uid_col: str = "uid",
        random_state: int = 42,
        keep_controls_if_no_cases: bool = True,
        proba_col: str = "prob",
        binary_group: str = "true_class"
    ):
        self.bin_edges = bin_edges if bin_edges is not None else np.linspace(0, 1, 11)
        self.bin_labels = [
            f"{a:.2f}-{b:.2f}" for a, b in zip(self.bin_edges[:-1], self.bin_edges[1:])
        ]
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.uid_col = uid_col
        self.keep_if_empty = keep_controls_if_no_cases
        self.random_state = random_state
        self.sampler: Optional[Any] = None
        self.binary_group = binary_group
        self.proba_col = proba_col
    def assign_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["risk_bin"] = pd.cut(
            df[self.proba_col],
            bins=self.bin_edges,
            labels=self.bin_labels,
            include_lowest=True,
            right=False,
        )
        return df

    def impute_features(
        self, df: pd.DataFrame, exclude_cols: List[str]
    ) -> pd.DataFrame:
        df = df.copy()
        X = df.drop(columns=exclude_cols)

        # ----- drop columns that are entirely NaN in this bin -----
        cols_all_nan = list(X.columns[X.isna().all()])
        numer_keep = X.drop(columns=cols_all_nan).select_dtypes(include=[np.number])
        print("Imputing on features: ",X.columns)
        # ———  Guard: if no numeric columns remain, just return the subset of df we need ———    
        if numer_keep.shape[1] == 0:
            print("Nothing numeric to impute, so just return the original df.")
            return df
        # ----- impute the remaining numeric block -----
        X_imp = pd.DataFrame(
            self.imputer.fit_transform(numer_keep),
            columns=numer_keep.columns,
            index=numer_keep.index,
        )
        for col in cols_all_nan:
            print("Only NaNs in ", col)
            X_imp[col] = -1.0
        for c in exclude_cols:
            X_imp[c] = df[c]
        return X_imp

    def match_case_control(
        self, df: pd.DataFrame, target_controls: int, target_cases: int, exclude_cols: List[str]=[],verbose=False
    ) -> pd.DataFrame:
        """
                1:k matching: keep all cases, then pick the 'target_controls' closest controls.
                If no cases:
        +          - when self.keep_if_empty==True: return all controls unchanged
        +          - else: return an empty DataFrame
        """
        cases = df[df[self.binary_group] == 1].reset_index(drop=True)
        controls = df[df[self.binary_group] == 0].reset_index(drop=True)
        # handle empty-case bins
        if len(cases) == 0:
                return df.sample(n=min(target_controls, len(controls)),
            random_state=self.random_state)
        if len(controls) == 0:
            return df.sample(
            n=min(target_cases, len(cases)),
            random_state=self.random_state)

        # 1) Define the *universe* of candidate covariates
        drop_cols = {self.uid_col, self.binary_group, self.proba_col, "risk_bin"} | set(exclude_cols)
        all_cols = [c for c in df.columns if c not in drop_cols]
        BIN_FLAG_COLUMNS = get_bin_flag_columns(df)
        CAT_COLUMNS = df.select_dtypes(include=["object","category"]).columns.tolist()
        TRUE_NUM_COLUMNS = df.select_dtypes(include=["int","float"]).columns.tolist()
        # 2) Intersect with your three feature pools
        cat_feats  = [c for c in CAT_COLUMNS         if c in all_cols]
        bin_feats  = [c for c in BIN_FLAG_COLUMNS    if c in all_cols]
        num_feats  = [c for c in TRUE_NUM_COLUMNS    if c in all_cols and c not in bin_feats]

        if verbose:
            print(">>> Matching will use:")
            print("    Categorical features:", cat_feats)
            print("    Numeric   features:", num_feats)
            print("    Binary    features :", bin_feats)
            # 3) Build a preprocessor that OHEs cat+bin and scales num
        preprocessor = get_preprocessor(
            df =df,
            categorical_cols = cat_feats ,
            numeric_cols     = num_feats
        )
        # fit on the combined set so that OHE sees all categories
        preprocessor.fit(pd.concat([cases, controls], ignore_index=True)[all_cols])

        # now transform each
        X_cases    = preprocessor.transform(cases)
        X_controls = preprocessor.transform(controls)   
        # grab all output names:
        feat_names = preprocessor.get_feature_names_out()

        if verbose: print([n for n in feat_names if n.endswith(tuple(BIN_FLAG_COLUMNS))])

        # -------- distance: control → nearest case ----------------------
        nbrs_cases  = NearestNeighbors(
            n_neighbors=min(len(cases), len(controls))).fit(X_cases)
        d_ctrl, _   = nbrs_cases.kneighbors(X_controls)
        min_d_ctrl  = d_ctrl.min(axis=1)          # length == len(controls)

        # keep the controls with the *smallest* of these distances
        pick_idx_ctrl = np.argsort(min_d_ctrl)[:min(target_controls, len(controls))]
        matched_ctrl  = controls.iloc[pick_idx_ctrl]

        # -------- distance: case → nearest control ----------------------
        nbrs_ctrl   = NearestNeighbors(
            n_neighbors=min(len(cases), len(controls))).fit(X_controls)
        d_case, _   = nbrs_ctrl.kneighbors(X_cases)
        min_d_case  = d_case.min(axis=1)          # length == len(cases)

        pick_idx_case = np.argsort(min_d_case)[:min(target_cases, len(cases))]
        matched_case  = cases.iloc[pick_idx_case]

        # -------- assemble & shuffle ------------------------------------
        matched = (
            pd.concat([matched_case, matched_ctrl], ignore_index=True)
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )
        return matched.copy()

    """ 
    Tuning Parameters:

    global_ctrl_ratio gives a fixed case:control ratio in every bin.

    target_bin_quantile equalizes total row counts across bins (good for balancing extremely sparse vs. dense strata). Shrink / stretch every bin toward the same target size.

    max_ctrl_per_case is a safety brake you can combine with either of the above to prevent runaway bin sizes. Put an upper cap on k so huge bins don't explode.
    """
    def apply_sampler_by_bin(self, df: pd.DataFrame,target_bin_quantile: Union[float, None] = None, 
    global_ctrl_ratio: Union[float, None] = None,max_ctrl_per_case: Union[float, None] = None):
      
        # 0) choose a target size once, if the user asked for a quantile
        if target_bin_quantile is not None:
            target_bin_size = int(
                np.percentile(df["risk_bin"].value_counts(), target_bin_quantile)
            )
        else:
            target_bin_size = int(df["risk_bin"].value_counts().mean())

        removed_ids, parts = {}, []
        # drop any non-numeric before median-impute
        non_numeric = df.select_dtypes(include=["object","category"]).columns.tolist()
        do_not_impute = non_numeric + [ self.uid_col, self.binary_group, self.proba_col ] #risk_bin 
        print("→ do_not_impute on :", do_not_impute)


        for b, bin_df in df.groupby("risk_bin", dropna=True):
           # print(f"\n--- bin {b} (n={len(bin_df)}) ---")

            imp_df = self.impute_features(bin_df, exclude_cols=do_not_impute)
            n_cases = (imp_df[self.binary_group] == 1).sum()
            n_ctrl = (imp_df[self.binary_group] == 0).sum()

            # -- decide how many controls to keep --------------------------
            if global_ctrl_ratio is not None:
                n_keep_ctrl = int(global_ctrl_ratio * n_cases)
                n_keep_case = int(global_ctrl_ratio * n_ctrl)  # by design for 1:k matching

            else:
                n_keep_ctrl = max(0, target_bin_size - n_cases)
                n_keep_case = min(n_cases, n_keep_ctrl)

            if max_ctrl_per_case is not None:
                n_keep_ctrl = min(n_keep_ctrl, max_ctrl_per_case * n_cases)


            n_keep_ctrl = min(n_keep_ctrl, n_ctrl)  # cannot keep more than we have

            matched = self.match_case_control(imp_df, target_controls=n_keep_ctrl, target_cases = n_keep_case)
            # --------------------------------------------------------------

            removed_ids[b] = list(
                set(bin_df[self.uid_col]) - set(matched[self.uid_col])
            )
            parts.append(matched.assign(risk_bin=b))

        return pd.concat(parts).reset_index(drop=True), removed_ids

