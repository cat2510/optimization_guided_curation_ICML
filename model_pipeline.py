from sklearn import calibration
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix,precision_recall_curve

def get_bin_flag_columns(df):
    return [col for col in df.columns if col.startswith("has_") or "THRCLS" in col.upper()
    or col.endswith("adherent") or col.startswith("early_")
    or col.startswith("is_")]
def get_true_num_columns(df, CAT_COLUMNS):
    return [
        col for col in df.columns
        if (
            ("cost" in col.lower() or 
             "quarterly" in col.lower() or 
             "claims" in col.lower())
            and col not in CAT_COLUMNS
        )
    ]

def get_preprocessor(df,categorical_cols, numeric_cols,verbose: bool = True,
):
    """
    Returns a ColumnTransformer that one-hot-encodes categorical columns
    and scales numeric columns.  If verbose=True, prints which cols go where.
    """
    # Filter columns to those present in df
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    if verbose:
        print("→ Building preprocessor:")
        print(f"   • OneHotEncoder on: {categorical_cols}")
        print(f"   • StandardScaler on: {numeric_cols}")

    preprocessor = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor

def get_stratifier_model(df, categorical_cols, numeric_cols, model_type='xgb_multiclass', 
                       preprocessor=None):
    """
    Returns a sklearn Pipeline for risk stratification.
    
    Args:
        df: DataFrame for preprocessing setup
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        model_type: 'regression', 'binary', 'xgb_multiclass'
        preprocessor: Optional preprocessor, will create if None
        n_strata: Number of cost strata for multiclass classification
    
    Returns:
        sklearn Pipeline and a function to convert predictions to risk scores
    """
    from sklearn.ensemble import RandomForestRegressor
    from statsmodels.distributions.empirical_distribution import ECDF
    import xgboost as xgb
    
    if preprocessor is None:
        preprocessor = get_preprocessor(df, categorical_cols, numeric_cols, verbose=False)
    
    if model_type == 'regression':
        # Original regression approach
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        def convert_to_risk_scores(predictions, training_predictions=None):
            """Convert continuous predictions to risk percentiles using ECDF"""
            if training_predictions is not None:
                ecdf_func = ECDF(training_predictions)
            else:
                ecdf_func = ECDF(predictions)
            return ecdf_func(predictions)
    
    elif model_type == 'xgb_multiclass':
        # XGBoost multi-class classifier for cost strata
        model = xgb.XGBClassifier(
            objective='multi:softprob',  # Multi-class with probabilities
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # For multi-class, we don't need risk score conversion
        def convert_to_risk_scores(predictions, training_predictions=None):
            return None  # Not needed for multi-class
    
    else:
        raise ValueError("model_type must be 'regression' or 'xgb_multiclass'")
    
    # Build pipeline
    steps = [
        ("preprocessor", preprocessor),
        ("model", model)
    ]
    
    pipeline = Pipeline(steps)
    
    return pipeline, convert_to_risk_scores

def get_histgb_pipeline(df,categorical_cols, numeric_cols,preprocessor=None, n_estimators=200, max_iter=200,
                                random_state=42, n_jobs=-1, calibration_method=None,balance_classes=True):

    if preprocessor is None:
            preprocessor = get_preprocessor(df=df,categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    # 2) wrap a gradient booster in a calibrator to get *better-spread* probabilities
    gb_kwargs = dict(max_iter=max_iter, random_state=random_state)
    if balance_classes:
        print("Class weights balanced for XGBoost model...")
        gb_kwargs['class_weight'] = "balanced"
    gb = HistGradientBoostingClassifier(**gb_kwargs)
    if calibration_method:
        print("Scaling prediction probabilities with CalibratedClassifierCV " + calibration_method)
        calibrated_gb = CalibratedClassifierCV(gb, cv=5, method=calibration_method) # other methods: "isotonic", "sigmoid"
    else:
        calibrated_gb = gb
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_gb)
    ])

    return pipeline

def get_logistic_pipeline(df,categorical_cols, numeric_cols,preprocessor=None, C=1.0, max_iter=500, class_weight=None,scale_post_preprocess=True,calibrate=False):

    """
    Returns a sklearn Pipeline with a preprocessor and logistic regression.
    """
    if preprocessor is None:
        preprocessor = get_preprocessor(df,categorical_cols, numeric_cols)
    logreg = LogisticRegression(
        solver="lbfgs",
        #multi_class="multinomial",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=42
    )

    steps = [("preprocessor", preprocessor)]
    
    if scale_post_preprocess:
        steps.append(("post_scaler", StandardScaler()))
    
    if calibrate:
        print("Calibrated Logistic Stratifier...")
        clf_cal = CalibratedClassifierCV(logreg, method="isotonic", cv=5)
        steps.append(("classifier", clf_cal))

    else:
        steps.append(("classifier", logreg))


    pipeline = Pipeline(steps)
    return pipeline


def get_random_forest_pipeline(df,categorical_cols, numeric_cols,preprocessor=None, n_estimators=200, max_depth=None,
                                min_samples_leaf=1, class_weight=None, n_jobs=-1):
    """
    Returns a sklearn Pipeline with a preprocessor and random forest classifier.
    """
    if preprocessor is None:
        preprocessor = get_preprocessor(df,categorical_cols, numeric_cols)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42,
        n_jobs=n_jobs
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])
    return pipeline


def train_test_split_enrol(df, target_col="high_cost_2018", test_size=0.3, random_state=42,verbose=True):
    """
    Splits df by ENROLID into train/test, stratifying on target_col.
    Returns: train_df, test_df
    """
    # Ensure ENROLID is unique per row
    assert df["ENROLID"].is_unique, "DataFrame must have one row per ENROLID"

    # Stratified split on ENROLID
    train_ids, test_ids = train_test_split(
        df["ENROLID"],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    train = df[df["ENROLID"].isin(train_ids)].reset_index(drop=True)
    test  = df[df["ENROLID"].isin(test_ids)].reset_index(drop=True)
    if verbose:
        # Debug prints
        print("Train/Test shapes:", train.shape, test.shape)
        print("Train distribution of {}:".format(target_col))
        print(train[target_col].value_counts(normalize=True))
        print("Test distribution of {}:".format(target_col))
        print(test[target_col].value_counts(normalize=True))

    return train_ids, test_ids, train, test



def evaluate_model_auc(clf, X_test, y_test, optimal_threshold=True):
    """
    Evaluates a fitted classifier pipeline by printing ROC AUC.
    Confusion matrix + classification report done with F1-optimized thresholding.
    """

    # True labels and predicted
    y_test = np.asarray(y_test)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    if optimal_threshold:
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        threshold = best_threshold
    else:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"Binary AUC: {auc:.3f}")

    # classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
