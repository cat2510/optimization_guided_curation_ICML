# -----------------------------------------------------------------------------
# IAI OPTIMAL CLASSIFICATION TREES INTEGRATION
# -----------------------------------------------------------------------------
# Add to your existing model_pipeline.py file
# -----------------------------------------------------------------------------

import pandas as pd
try:
    from interpretableai import iai
    IAI_AVAILABLE = True
except ImportError:
    print("Warning: interpretableai not installed. IAI models will not be available.")
    IAI_AVAILABLE = False
from sklearn.metrics import classification_report, confusion_matrix
from model_pipeline import get_preprocessor, train_test_split_enrol

def train_opt_with_feature_names(X_train, treatments, outcomes,
                                 categorical_cols, numeric_cols,
                                 max_depth=5, minbucket=50, cp=0.001):
   
    # Step 1: Create and fit preprocessor
    from model_pipeline import get_preprocessor
    preprocessor = get_preprocessor(X_train, categorical_cols, numeric_cols)
    
    # Step 2: Fit and transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Step 3: Get feature names after transformation
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'ohe':
            # OneHotEncoder - get encoded feature names
            ohe_features = transformer.get_feature_names_out(columns)
            feature_names.extend(ohe_features)
        elif name == 'num':
            # StandardScaler - keeps same names
            feature_names.extend(columns)
        elif name == 'remainder':
            # Passthrough features
            if preprocessor.remainder == 'passthrough':
                # Get columns not in other transformers
                all_cols = X_train.columns.tolist()
                used_cols = []
                for _, _, cols in preprocessor.transformers_[:-1]:
                    used_cols.extend(cols)
                remainder_cols = [c for c in all_cols if c not in used_cols]
                feature_names.extend(remainder_cols)
    
    # Step 4: Create DataFrames with proper feature names
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    (train_X, train_treatments, train_outcomes), (test_X, test_treatments, test_outcomes) = (
        iai.split_data('policy_maximize', X_train_df, treatments, outcomes, seed=123, train_proportion=0.5))
    reward_lnr = iai.CategoricalClassificationRewardEstimator(
        propensity_estimator=iai.RandomForestClassifier(),
        outcome_estimator=iai.RandomForestClassifier(),
        reward_estimator='direct_method',
        random_seed=123,
    )
    train_predictions, train_reward_score = reward_lnr.fit_predict(
        train_X, train_treatments, train_outcomes,
        propensity_score_criterion='auc', outcome_score_criterion='auc')
    train_rewards = train_predictions['reward']
    grid = iai.GridSearch(
    iai.OptimalTreePolicyMaximizer(
        random_seed=121,
        max_categoric_levels_before_warning=20,
    ),
    max_depth=range(6),
    )
    grid.fit(train_X, train_rewards)
    opt_learner = grid.get_learner()
    
    return opt_learner, preprocessor, feature_names


def train_oct_with_feature_names(X_train, y_train, 
                                 categorical_cols, numeric_cols,
                                 max_depth=5, minbucket=50, cp=0.001):
    """
    Train IAI with proper feature names by transforming data first
    
    This is the RECOMMENDED approach - transform first, then train IAI directly
    """
    
    # Step 1: Create and fit preprocessor
    preprocessor = get_preprocessor(X_train, categorical_cols, numeric_cols)
    
    # Step 2: Fit and transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Step 3: Get feature names after transformation
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'ohe':
            # OneHotEncoder - get encoded feature names
            ohe_features = transformer.get_feature_names_out(columns)
            feature_names.extend(ohe_features)
        elif name == 'num':
            # StandardScaler - keeps same names
            feature_names.extend(columns)
        elif name == 'remainder':
            # Passthrough features
            if preprocessor.remainder == 'passthrough':
                # Get columns not in other transformers
                all_cols = X_train.columns.tolist()
                used_cols = []
                for _, _, cols in preprocessor.transformers_[:-1]:
                    used_cols.extend(cols)
                remainder_cols = [c for c in all_cols if c not in used_cols]
                feature_names.extend(remainder_cols)
    
    # Step 4: Create DataFrames with proper feature names
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    
    # Step 5: Train IAI directly (no pipeline needed)
    
    iai_model = iai.OptimalTreeClassifier(
        max_depth=max_depth,
        minbucket=minbucket,
        cp=cp,
        random_seed=42
    )
    
    iai_model.fit(X_train_df, y_train)
    
     
    return iai_model, preprocessor, feature_names


import itertools
from sklearn.metrics import precision_score, recall_score, f1_score

def finetune_oct(X_train, y_train, X_val, y_val, categorical_cols, numeric_cols,
                 depths=[5, 7,9],
                 minbuckets=[50, 100,150],
                 cps=[0.001, 0.01, 0.05]):
    """
    Hyperparameter tuning for IAI OptimalTreeClassifier
    Selects best hyperparameters based on F1 score
    """

    best_score = -1
    best_params = None
    best_model = None
    results = []
    
    preprocessor = get_preprocessor(X_train, categorical_cols, numeric_cols)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    # Get feature names
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'ohe':
            ohe_features = transformer.get_feature_names_out(columns)
            feature_names.extend(ohe_features)
        elif name == 'num':
            feature_names.extend(columns)
        elif name == 'remainder' and preprocessor.remainder == 'passthrough':
            all_cols = X_train.columns.tolist()
            used_cols = []
            for _, _, cols in preprocessor.transformers_[:-1]:
                used_cols.extend(cols)
            remainder_cols = [c for c in all_cols if c not in used_cols]
            feature_names.extend(remainder_cols)

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_transformed, columns=feature_names)

    # Grid search
    for depth, minbucket, cp in itertools.product(depths, minbuckets, cps):

        model = iai.OptimalTreeClassifier(
            max_depth=depth,
            minbucket=minbucket,
            cp=cp,
            random_seed=123
        )
        model.fit(X_train_df, y_train)

        y_pred = model.predict(X_val_df)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        results.append({
            "depth": depth,
            "minbucket": minbucket,
            "cp": cp,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        if f1 > best_score:
            best_score = f1
            best_params = (depth, minbucket, cp)
            best_model = model

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)

    return best_model, best_params, results_df,preprocessor, feature_names

def evaluate_binary_oct(iai_model,X_test_df,y_test, preprocessor, feature_names):
    
    print(f"Test dataset for OCT application: {len(X_test_df):,} samples")
    
    try:
        # Transform full dataset using the same preprocessor
        X_test_processed = preprocessor.transform(X_test_df)
        print(f"✓ Test data preprocessing completed: {X_test_processed.shape}")
        print(f"✓ Feature names: {feature_names}")
        # Create DataFrame for leaf assignment
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
        # Get multi-class predictions (cost strata)
        y_pred = iai_model.predict(X_test_processed)
        leaf_assignments = iai_model.apply(X_test_processed)
        print(f"✓ Cost stratum test predictions completed")
    except Exception as e:
        print(f"✗ Error applying OCT to test dataset: {e}")
        raise e

    X_test_processed['leaf_assignment'] = leaf_assignments
    X_test_processed['predicted_cost_stratum'] = y_pred
    
    # Metrics
    auc = iai_model.score(X_test_processed, y_test, criterion='auc')
    misclassification_score= iai_model.score(X_test_processed, y_test, criterion='misclassification')

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    print(f"AUC score: {auc:.3f}")
    print(f"Misclassification score: {misclassification_score:.3f}")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    #display(iai_model.ROCCurve(X_test_processed, y_test,positive_label=1))

    return X_test_processed


def train_and_evaluate_for_w(
    matched_df,
    X_test,
    y_test,
    feature_cols,
    target_col='highcost_gt_200000',
    categorical_cols=None,
    numeric_cols=None,
    minbuckets=[50, 100, 120, 150],
    cps=[0.00001, 5e-4, 0.0001, 5e-3, 0.001, 0.01],
    depths=[5, 7, 9],
    verbose=True
):
    """Train model on matched_df and evaluate using your evaluate_binary_oct function"""
    
    # Split matched data
    _, _, train_df, val_df = train_test_split_enrol(
        matched_df, 
        target_col="cost_stratum_2018", # TODO: change to cost_stratum_2018
        test_size=0.3,
        verbose=False
    )
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    if verbose:
        print(f"Training: {len(X_train):,}, Validation: {len(X_val):,}")
    
    # Train model
    model, best_params, val_results, preprocessor, feature_names = finetune_oct(
        X_train, y_train, X_val, y_val,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        depths=depths,
        minbuckets=minbuckets,
        cps=cps
    )
    
    if verbose:
        print(f"Best params: {best_params}")
    
    # Evaluate on test using YOUR function
    X_test_processed = evaluate_binary_oct(
        model, X_test, y_test, 
        preprocessor, feature_names
    )
    
    # Extract metrics from the evaluation output
    y_pred = X_test_processed['predicted_cost_stratum']
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn)  # Sensitivity
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # AUC (get from model score)
    auc = model.score(X_test_processed.drop(columns=['leaf_assignment', 'predicted_cost_stratum']), 
                      y_test, criterion='auc')
    
    return {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'X_test_processed': X_test_processed,
        'auc': auc,
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'train_size': len(X_train),
        'val_size': len(X_val)
    }

