import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def get_feature_importance(model, X, y=None, method='built_in', n_repeats=10, random_state=42, top_n=None):
    """
    Extract feature importance from a trained model.
    
    Parameters:
    -----------
    model : sklearn estimator or pipeline
        Trained model from which to extract feature importance.
    X : Features used for training/testing the model.
    y : Target variable, required only for permutation importance.
    method : str, default='built_in'
        Method to use for feature importance:
        - 'built_in': Use model's built-in feature_importances_ or coef_ attribute
        - 'permutation': Use permutation importance (model-agnostic but more computationally expensive)
    n_repeats : int, default=10
        Number of times to permute each feature (only used if method='permutation').
    top_n : int, optional
        If provided, return only the top N most important features.
        
    Returns:
    --------
    DataFrame
        Feature names and their importance scores, sorted by importance.
    """
    # If model is a pipeline, get the final estimator
    if isinstance(model, Pipeline):
        pipeline = model
        model = pipeline.named_steps['classifier']
        
        # Get feature names after preprocessing (for models with preprocessors)
        # This assumes the pipeline has a 'preprocessor' step
        if 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
            # Check if the preprocessor has get_feature_names_out method (newer scikit-learn versions)
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            # Try to get feature names from transformers (older scikit-learn versions)
            else:
                try:
                    # Handle ColumnTransformer case
                    ohe = preprocessor.named_transformers_['ohe']
                    numeric = preprocessor.named_transformers_['num']
                    cat_features = preprocessor.transformers_[0][2]
                    num_features = preprocessor.transformers_[1][2]
                    
                    # Get categorical feature names after one-hot encoding
                    if hasattr(ohe, 'get_feature_names_out'):
                        cat_feature_names = ohe.get_feature_names_out(cat_features)
                    else:
                        cat_feature_names = [f"{col}_{val}" for col in cat_features 
                                            for val in ohe.categories_[i][1:] 
                                            for i, col in enumerate(cat_features)]
                    
                    # Combine with numeric feature names
                    feature_names = np.concatenate([cat_feature_names, num_features])
                except:
                    # If we can't determine feature names, use indices
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            # If no preprocessor, use original feature names
            feature_names = X.columns.tolist()
    else:
        # If not a pipeline, use original feature names
        feature_names = X.columns.tolist()
    
    # Extract importance using the specified method
    if method == 'built_in':
        # For models with feature_importances_ attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # For models with coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            # Handle multi-class case
            if len(model.coef_.shape) > 1:
                # Use absolute value of coefficients and average across classes
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
        # Handle special case for CalibratedClassifierCV
        elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'feature_importances_'):
            importances = model.base_estimator.feature_importances_
        elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'coef_'):
            if len(model.base_estimator.coef_.shape) > 1:
                importances = np.mean(np.abs(model.base_estimator.coef_), axis=0)
            else:
                importances = np.abs(model.base_estimator.coef_)
        else:
            raise ValueError(f"Model {type(model).__name__} doesn't support built-in feature importance. "
                           "Try method='permutation' instead.")
    
    elif method == 'permutation':
        if y is None:
            raise ValueError("Target variable 'y' must be provided for permutation importance.")
        
        if not isinstance(X, np.ndarray) and X.select_dtypes(include=['object', 'category']).shape[1] > 0:
            print("Warning: Categorical variables detected. Converting to numeric representation for permutation importance.")
            # Convert categorical columns to numeric
            X_numeric = X.copy()
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X_numeric[col] = pd.factorize(X[col])[0]
            X_perm = X_numeric
        else:
            X_perm = X
        
        # Use pipeline for permutation to include preprocessing steps
        if isinstance(model, Pipeline):
            try:
                importances = permutation_importance(
                    pipeline, X_perm, y, n_repeats=n_repeats, random_state=random_state
                )
            except Exception as e:
                print(f"Error with pipeline permutation: {e}")
                print("Trying alternative approach with pre-processed data...")
                # Alternative approach: apply preprocessing first, then compute permutation importance on the transformed data
                X_transformed = pipeline.named_steps['preprocessor'].transform(X_perm)
                importances = permutation_importance(
                    pipeline.named_steps['classifier'], X_transformed, y, 
                    n_repeats=n_repeats, random_state=random_state
                )
        else:
            importances = permutation_importance(
                model, X_perm, y, n_repeats=n_repeats, random_state=random_state
            )
      
    # Create DataFrame with feature names and importance scores
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Limit to top_n features if specified
    if top_n is not None and top_n < len(importance_df):
        importance_df = importance_df.head(top_n)
    
    return importance_df


def plot_feature_importance(importance_df, title="Feature Importance", 
                           figsize=(12, 8), color_palette="viridis", 
                           orientation='horizontal', show_values=True, 
                           value_format=".3f"):
    """
    Plot feature importance from a trained model.
    
    Parameters:
    -----------
    importance_df : DataFrame
        DataFrame with feature importance values, output from get_feature_importance().
    title : str, default="Feature Importance"
        Title for the plot.
    figsize : tuple, default=(12, 8)
        Figure size.
    color_palette : str, default="viridis"
        Color palette for the bars.
    orientation : str, default='horizontal'
        Orientation of the bar plot ('horizontal' or 'vertical').
    show_values : bool, default=True
        Whether to show importance values on the bars.
    value_format : str, default=".3f"
        Format string for the importance values.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object.
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get number of features to plot
    n_features = len(importance_df)
    
    # Create colormap
    colors = sns.color_palette(color_palette, n_features)
    
    # Reverse order for horizontal orientation (to show most important at the top)
    if orientation == 'horizontal':
        plot_df = importance_df.iloc[::-1].copy()
    else:
        plot_df = importance_df.copy()
    
    # Plot
    if orientation == 'horizontal':
        bars = ax.barh(plot_df['feature'], plot_df['importance'], color=colors)
        ax.set_xlabel('Importance')
        ax.set_ylabel('')
    else:
        bars = ax.bar(plot_df['feature'], plot_df['importance'], color=colors)
        ax.set_xlabel('')
        ax.set_ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
    
    # Add values on bars
    if show_values:
        for bar in bars:
            if orientation == 'horizontal':
                width = bar.get_width()
                ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                       f"{width:{value_format}}", 
                       va='center')
            else:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height * 1.01,
                       f"{height:{value_format}}",
                       ha='center', va='bottom', rotation=0)
    
    # Set title
    ax.set_title(title, fontsize=14, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    
    return fig


def analyze_feature_importance(clf, X_train, y_train=None, X_test=None, y_test=None, 
                              method='built_in', top_n=20, title=None, 
                              figsize=(12, 8), orientation='horizontal', 
                               return_df=False):
    """
    Combined function to extract and plot feature importance.
    
    Parameters:
    -----------
    clf : sklearn estimator or pipeline
        Trained model from which to extract feature importance.
    method : str, default='built_in'
        Method to use for feature importance ('built_in' or 'permutation').
    top_n : int, default=20
        Number of top features to display.
    title : str, optional
        Title for the plot. If None, a default title is generated.
    figsize : tuple, default=(12, 8)
        Figure size.
    orientation : str, default='horizontal'
        Orientation of the bar plot ('horizontal' or 'vertical').
    save_path : str, optional
        If provided, save the plot to this path.
    return_df : bool, default=False
        If True, return the feature importance DataFrame.
        
    Returns:
    --------
    matplotlib.figure.Figure or tuple
        The created figure object, or a tuple (fig, importance_df) if return_df=True.
    """
    # Set default title based on method and model type
    if title is None:
        model_name = type(clf).__name__
        if isinstance(clf, Pipeline):
            if hasattr(clf, 'named_steps') and 'classifier' in clf.named_steps:
                model_name = type(clf.named_steps['classifier']).__name__
        
        title = f"{model_name} Feature Importance ({method.capitalize()} Method)"
    
    # Determine data to use for permutation importance
    X_perm = X_test if X_test is not None else X_train
    y_perm = y_test if y_test is not None else y_train
    
    # Get feature importance
    importance_df = get_feature_importance(
        clf, X_perm, y_perm, method=method, top_n=top_n
    )
    
    # Plot feature importance
    fig = plot_feature_importance(
        importance_df, title=title, figsize=figsize, 
        orientation=orientation
    )
    
    if return_df:
        return fig, importance_df
    else:
        return fig