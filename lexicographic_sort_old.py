import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
# Import the matching class from the other file
from balancing_functions.optimal_match_control import EnhancedRiskBinnedCaseControlResampler  # Adjust import path as needed
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import time
def sort_features_by_density(X, feature_names):
    """
    Sort features from most dense (fewest zeros) to most sparse (most zeros)
    
    Parameters:
    -----------
    X : np.array
        Feature matrix
    feature_names : list
        Feature column names
        
    Returns:
    --------
    list : Column indices sorted by density (most dense first)
    """
    density_scores = []
    for col_idx in range(X.shape[1]):
        non_zero_count = np.sum(X[:, col_idx] != 0)
        density_score = non_zero_count / X.shape[0]  # Proportion of non-zero values
        density_scores.append((density_score, col_idx, feature_names[col_idx]))
    
    # Sort by density (descending - most dense first)
    sorted_features = sorted(density_scores, key=lambda x: x[0], reverse=True)
    
    return [x[1] for x in sorted_features]  # Return column indices

def lexicographic_sort_patients(X, sorted_feature_indices):
    """
    Sort patients lexicographically by feature density
    - First by most dense feature (non-zero comes before zero)  
    - Then by second most dense feature, etc.
    
    Parameters:
    -----------
    X : np.array
        Feature matrix
    sorted_feature_indices : list
        Feature indices sorted by density
        
    Returns:
    --------
    list : Patient indices sorted lexicographically
    """
    sort_keys = []
    for patient_idx in range(X.shape[0]):
        # Create lexicographic key: tuple of (is_nonzero_feat1, is_nonzero_feat2, ...)
        # Use negative values so non-zero (True=-1) comes before zero (False=0)
        key = tuple(-int(X[patient_idx, feat_idx] != 0) 
                   for feat_idx in sorted_feature_indices)
        sort_keys.append((key, patient_idx))
    
    # Sort lexicographically and return patient indices
    sorted_patients = sorted(sort_keys, key=lambda x: x[0])
    return [x[1] for x in sorted_patients]

def undersample_majority_class_lexicographic(leaf_df, target_col, feature_cols, matcher_instance,
                                                     undersampling_ratio=1.0,
                                                     top_k_factor=8.0,
                                                     min_majority_samples=50,
                                                     verbose=False):
    """
    Undersample majority class using lexicographic ordering + distance-based selection
    
    Strategy:
    1. Lexicographically sort both majority and minority classes
    2. Only compute distances among top-K sorted patients from each class  
    3. Select majority samples that are closest to minority samples
    
    Parameters:
    -----------
    leaf_df : pd.DataFrame
        Data for a single OCT leaf
    target_col : str
        Target column name
    feature_cols : list
        Feature columns for sorting and distance computation
    undersampling_ratio : float
        Ratio of majority:minority after undersampling
    top_k_factor : float
        Compute distances only among top (target_majority * top_k_factor) sorted patients
    min_majority_samples : int
        Minimum majority samples to keep
    verbose : bool
        Print detailed information
        
    Returns:
    --------
    pd.DataFrame : Undersampled dataframe
    """
    
    if len(leaf_df) == 0:
        return leaf_df.copy()
    
    # Identify majority and minority classes
    class_counts = leaf_df[target_col].value_counts()
    if len(class_counts) < 2:
        return leaf_df.copy()
    
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[1]
    
    n_majority = class_counts[majority_class] 
    n_minority = class_counts[minority_class]
    
    # Calculate target number of majority samples
    target_majority = max(int(n_minority * undersampling_ratio), min_majority_samples)
    
    if target_majority >= n_majority:
        return leaf_df.copy()
    
    # Separate classes
    majority_df = leaf_df[leaf_df[target_col] == majority_class].copy()
    minority_df = leaf_df[leaf_df[target_col] == minority_class].copy()
    
    # Prepare features
    available_features = [col for col in feature_cols if col in majority_df.columns]
    if len(available_features) == 0:
        sampled_majority = majority_df.sample(n=target_majority, random_state=42)
        return pd.concat([sampled_majority, minority_df], ignore_index=True)
    
    # Use the same preprocessing pipeline as your optimization matching
    # This handles categorical encoding, normalization, etc.
    try:
        # Use your existing preprocessing method
        exclude_cols_matching = [col for col in majority_df.columns 
                               if col not in feature_cols or col in [target_col, 'leaf_assignment']]
        
        X_majority, X_minority = matcher_instance.get_preprocessed_control_case_features(
            cases=majority_df, 
            controls=minority_df,
            exclude_cols_matching=exclude_cols_matching,
            verbose=verbose
        )
        
        # Get feature names after preprocessing (will be different due to one-hot encoding)
        processed_feature_count = X_majority.shape[1]
        processed_feature_names = [f"feature_{i}" for i in range(processed_feature_count)]
        
        if verbose:
            print(f"    Preprocessing: {len(feature_cols)} → {processed_feature_count} features")
        
    except Exception as e:
        if verbose:
            print(f"    Preprocessing failed: {e}, falling back to simple sampling")
        sampled_majority = majority_df.sample(n=target_majority, random_state=42)
        return pd.concat([sampled_majority, minority_df], ignore_index=True)
 
    # Sort features by density (combined from both classes for consistency)
    X_combined = np.vstack([X_majority, X_minority])
    sorted_feature_indices = sort_features_by_density(X_combined, processed_feature_names)
    
    # Lexicographically sort patients within each class
    majority_sorted_indices = lexicographic_sort_patients(X_majority, sorted_feature_indices)
    minority_sorted_indices = lexicographic_sort_patients(X_minority, sorted_feature_indices)
    
    # Determine top-K for distance computation (limit search space)
    top_k_majority = min(int(target_majority * top_k_factor), len(majority_sorted_indices))
    top_k_minority = len(minority_sorted_indices)  # Use all minority samples
    
    if verbose:
        print(f"  Distance computation scope:")
        print(f"    Top majority candidates: {top_k_majority} (from {n_majority})")
        print(f"    All minority samples: {top_k_minority}")
        print(f"    Total distance computations: {top_k_majority * top_k_minority}")
    
    # Extract top-K sorted patients for distance computation
    top_majority_indices = majority_sorted_indices[:top_k_majority]
    top_minority_indices = minority_sorted_indices[:top_k_minority]
    
    X_majority_topk = X_majority[top_majority_indices]
    X_minority_topk = X_minority[top_minority_indices]
    
    # Compute distances only among the lexicographically sorted top patients
    distances = cdist(X_majority_topk, X_minority_topk, metric='euclidean')
    
    if verbose:
        print(f"    Distance matrix shape: {distances.shape}")
        print(f"    Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    
    # For each majority sample, find its minimum distance to any minority sample
    min_distances_to_minority = np.min(distances, axis=1)
    
    # Select the target_majority samples with smallest distances to minority class
    closest_indices_in_topk = np.argsort(min_distances_to_minority)[:target_majority]
    
    # Map back to original majority dataframe indices
    selected_majority_indices = [top_majority_indices[i] for i in closest_indices_in_topk]
    sampled_majority = majority_df.iloc[selected_majority_indices].copy()
    
    if verbose:
        selected_distances = min_distances_to_minority[closest_indices_in_topk]
        print(f"  Selected majority samples:")
        print(f"    Average distance to minority: {selected_distances.mean():.3f}")
        print(f"    Distance std: {selected_distances.std():.3f}")
        
        # Compare with simple lexicographic selection
        simple_lex_indices = majority_sorted_indices[:target_majority]
        simple_distances = np.min(cdist(X_majority[simple_lex_indices], X_minority, metric='euclidean'), axis=1)
        print(f"  Comparison with simple lexicographic:")
        print(f"    Simple lex avg distance: {simple_distances.mean():.3f}")
        print(f"    Improvement: {((simple_distances.mean() - selected_distances.mean()) / simple_distances.mean() * 100):.1f}%")
    
    # Combine selected majority with all minority samples
    result_df = pd.concat([sampled_majority, minority_df], ignore_index=True)
    
    return result_df

def undersample_all_oct_leaves(df_with_strata_and_leaves, target_col, feature_cols,
                              leaf_col='leaf_assignment',
                              undersampling_ratio=1.0,
                              min_samples_per_leaf=10,
                              verbose=True):
    """
    Apply lexicographic undersampling to all OCT leaves
    
    Parameters:
    -----------
    df_with_strata_and_leaves : pd.DataFrame
        Full dataset with OCT leaf assignments
    target_col : str
        Target column for class imbalance
    feature_cols : list
        Features to use for lexicographic sorting
    leaf_col : str
        Column containing leaf assignments
    undersampling_ratio : float
        Target majority:minority ratio after undersampling
    min_samples_per_leaf : int
        Skip leaves with fewer samples than this
    verbose : bool
        Print progress information
        
    Returns:
    --------
    pd.DataFrame : Undersampled dataset
    """
    
    print(f"=== LEXICOGRAPHIC UNDERSAMPLING ACROSS OCT LEAVES ===")
    print(f"Original dataset: {len(df_with_strata_and_leaves):,} samples")
    
    # Get leaf assignments
    unique_leaves = df_with_strata_and_leaves[leaf_col].unique()
    print(f"Number of OCT leaves: {len(unique_leaves)}")
    
    undersampled_parts = []
    skipped_leaves = 0
    total_original = 0
    total_undersampled = 0
    # Initialize matcher instance
    matcher = EnhancedRiskBinnedCaseControlResampler(
    matching_method="ortools",  # Use OR-Tools for optimization
    random_state=42,
    binary_group="target_col",       # Your binary target column
    uid_col="ENROLID"            # Your ID column
    )
    for leaf_id in sorted(unique_leaves):
        leaf_df = df_with_strata_and_leaves[
            df_with_strata_and_leaves[leaf_col] == leaf_id
        ].copy()
        
        # Skip very small leaves
        if len(leaf_df) < min_samples_per_leaf:
            skipped_leaves += 1
            if verbose:
                print(f"Leaf {leaf_id}: Skipped (only {len(leaf_df)} samples)")
            continue
        
        total_original += len(leaf_df)
        
        # Apply lexicographic undersampling to this leaf
        if verbose:
            print(f"\nLeaf {leaf_id}: Processing {len(leaf_df)} samples...")
        
        undersampled_leaf = undersample_majority_class_lexicographic(
            leaf_df, 
            target_col='high_cost_2018',
            feature_cols=feature_cols,
            matcher_instance=matcher,
            undersampling_ratio=1.0,
            top_k_factor=3.0,  # Only compute distances for top 3x candidates
            verbose=True
        )
        total_undersampled += len(undersampled_leaf)
        undersampled_parts.append(undersampled_leaf)
        
        if verbose:
            reduction_pct = (1 - len(undersampled_leaf)/len(leaf_df)) * 100
            print(f"Leaf {leaf_id}: {len(leaf_df)} → {len(undersampled_leaf)} "
                  f"({reduction_pct:.1f}% reduction)")
    
    if len(undersampled_parts) == 0:
        print("Warning: No leaves processed successfully")
        return pd.DataFrame()
    
    # Combine all undersampled leaves
    final_undersampled = pd.concat(undersampled_parts, ignore_index=True)
    
    # Summary statistics
    print(f"\n=== UNDERSAMPLING SUMMARY ===")
    print(f"Leaves processed: {len(undersampled_parts)}")
    print(f"Leaves skipped: {skipped_leaves}")
    print(f"Overall reduction: {total_original:,} → {total_undersampled:,} "
          f"({(1-total_undersampled/total_original)*100:.1f}%)")
    
    # Class distribution before/after
    if target_col in df_with_strata_and_leaves.columns:
        print(f"\nClass distribution:")
        original_dist = df_with_strata_and_leaves[target_col].value_counts().sort_index()
        final_dist = final_undersampled[target_col].value_counts().sort_index()
        
        for class_val in original_dist.index:
            orig_count = original_dist.get(class_val, 0)
            final_count = final_dist.get(class_val, 0)
            print(f"  Class {class_val}: {orig_count:,} → {final_count:,}")
    
    return final_undersampled



# Integration with your pipeline
def create_undersampled_training_data(pipeline_results, target_col, 
                                     undersampling_ratio=1.0, verbose=True):
    """
    Create undersampled training data using the lexicographic approach
    
    Parameters:
    -----------
    pipeline_results : dict
        Results from create_cost_strata_pipeline()
    target_col : str
        Target column for classification (e.g., 'high_cost_2018')
    undersampling_ratio : float
        Target ratio of majority:minority class (1.0 = balanced)
    verbose : bool
        Print detailed information
        
    Returns:
    --------
    pd.DataFrame : Undersampled training dataset
    """
    
    df_with_leaves = pipeline_results['df_with_strata_and_leaves']
    feature_cols = pipeline_results['stratification_cols']
    
    # Apply lexicographic undersampling across all leaves
    undersampled_data = undersample_all_oct_leaves(
        df_with_leaves, 
        target_col=target_col,
        feature_cols=feature_cols,
        undersampling_ratio=undersampling_ratio,
        verbose=verbose
    )
    
    return undersampled_data


########################################################
# ############################
#  Comparison with full IP and example usage for smallest leaves:

def compare_matching_methods_in_leaf(leaf_df, target_col, feature_cols,
                                   matcher_instance, 
                                   undersampling_ratio=1.0,
                                   top_k_factor=3.0,
                                   min_majority_samples=50,
                                   verbose=True):
    """
    Compare lexicographic distance vs OR-Tools matching within a single OCT leaf
    
    Parameters:
    -----------
    leaf_df : pd.DataFrame
        Data for a single OCT leaf
    target_col : str
        Target column for class identification
    feature_cols : list
        Feature columns for matching
    matcher_instance : OptimalMatchControl
        Instance of your matching class with preprocessing methods
    undersampling_ratio : float
        Target majority:minority ratio
    top_k_factor : float
        For lexicographic method - search space multiplier
    min_majority_samples : int
        Minimum majority samples to keep
    verbose : bool
        Print comparison results
        
    Returns:
    --------
    dict : Comparison results with both methods' outputs and metrics
    """
    
    if len(leaf_df) == 0:
        return {'error': 'Empty leaf'}
    
    # Identify classes
    class_counts = leaf_df[target_col].value_counts()
    if len(class_counts) < 2:
        return {'error': 'Single class leaf'}
    
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[1]
    n_majority = class_counts[majority_class] 
    n_minority = class_counts[minority_class]
    
    target_majority = max(int(n_minority * undersampling_ratio), min_majority_samples)
    
    if target_majority >= n_majority:
        return {'error': 'No undersampling needed'}
    
    if verbose:
        print(f"\n=== LEAF MATCHING COMPARISON ===")
        print(f"Original: {n_majority} majority, {n_minority} minority")
        print(f"Target: {target_majority} majority (ratio {undersampling_ratio}:1)")
    
    # Separate classes
    majority_df = leaf_df[leaf_df[target_col] == majority_class].copy()
    minority_df = leaf_df[leaf_df[target_col] == minority_class].copy()
    
    exclude_cols = [target_col, 'leaf_assignment', 'cost_stratum'] + \
                   [col for col in leaf_df.columns if col not in feature_cols]
    
    results = {}
    
    # ===== METHOD 1: OR-TOOLS OPTIMIZATION =====
    print(f"\n--- Method 1: OR-Tools Optimization ---")
    start_time = time.time()
    
    try:
        ortools_matched = matcher_instance.ortools_optimization_matching(
            dfA=minority_df,  
            dfB=majority_df, 
            target_controls=target_majority, 
            target_cases=n_minority, 
            exclude_cols_matching=exclude_cols,
            verbose=False,
            max_controls_per_case=1
        )
        
        ortools_time = time.time() - start_time
        
        # Calculate total distance for OR-Tools result
        ortools_majority = ortools_matched[ortools_matched[target_col] == majority_class]
        ortools_minority = ortools_matched[ortools_matched[target_col] == minority_class]
        
        X_ortools_maj, X_ortools_min = matcher_instance.get_preprocessed_control_case_features(
            cases=ortools_majority, controls=ortools_minority, 
            exclude_cols_matching=exclude_cols, verbose=False
        )
        
        # Compute distances between all selected majority and minority samples
        ortools_distances = cdist(X_ortools_maj, X_ortools_min, metric='euclidean')
        ortools_total_distance = np.sum(np.min(ortools_distances, axis=1))  # Min distance per majority sample
        ortools_avg_distance = ortools_total_distance / len(ortools_majority)
        
        results['ortools'] = {
            'matched_data': ortools_matched,
            'n_majority_selected': len(ortools_majority),
            'n_minority_selected': len(ortools_minority),
            'total_distance': ortools_total_distance,
            'avg_distance_per_majority': ortools_avg_distance,
            'computation_time': ortools_time,
            'success': True
        }
        
        if verbose:
            print(f"✓ OR-Tools completed in {ortools_time:.2f}s")
            print(f"  Selected: {len(ortools_majority)} majority, {len(ortools_minority)} minority")
            print(f"  Total distance: {ortools_total_distance:.3f}")
            print(f"  Avg distance per majority: {ortools_avg_distance:.3f}")
            
    except Exception as e:
        results['ortools'] = {'success': False, 'error': str(e)}
        if verbose:
            print(f"✗ OR-Tools failed: {e}")
    
    # ===== METHOD 2: LEXICOGRAPHIC DISTANCE =====
    print(f"\n--- Method 2: Lexicographic Distance ---")
    start_time = time.time()
    
    try:
        # Use preprocessing from matcher instance
        X_majority, X_minority = matcher_instance.get_preprocessed_control_case_features(
            cases=majority_df, controls=minority_df,
            exclude_cols_matching=exclude_cols, verbose=False
        )
        
        processed_feature_count = X_majority.shape[1]
        processed_feature_names = [f"feature_{i}" for i in range(processed_feature_count)]
        
        # Sort features by density
        X_combined = np.vstack([X_majority, X_minority])
        sorted_feature_indices = sort_features_by_density(X_combined, processed_feature_names)
        
        # Lexicographically sort patients
        majority_sorted_indices = lexicographic_sort_patients(X_majority, sorted_feature_indices)
        minority_sorted_indices = lexicographic_sort_patients(X_minority, sorted_feature_indices)
        
        # Limit distance computation scope
        top_k_majority = min(int(target_majority * top_k_factor), len(majority_sorted_indices))
        
        top_majority_indices = majority_sorted_indices[:top_k_majority]
        top_minority_indices = minority_sorted_indices  # All minority
        
        X_majority_topk = X_majority[top_majority_indices]
        X_minority_topk = X_minority[top_minority_indices]
        
        # Compute distances
        distances = cdist(X_majority_topk, X_minority_topk, metric='euclidean')
        
        # Select majority samples closest to minority samples
        min_distances_to_minority = np.min(distances, axis=1)
        closest_indices_in_topk = np.argsort(min_distances_to_minority)[:target_majority]
        
        # Map back to original indices
        selected_majority_indices = [top_majority_indices[i] for i in closest_indices_in_topk]
        lex_majority_selected = majority_df.iloc[selected_majority_indices].copy()
        lex_minority_selected = minority_df.copy()  # All minority
        
        lex_matched = pd.concat([lex_majority_selected, lex_minority_selected], ignore_index=True)
        
        lex_time = time.time() - start_time
        
        # Calculate total distance for lexicographic result
        selected_distances = min_distances_to_minority[closest_indices_in_topk]
        lex_total_distance = np.sum(selected_distances)
        lex_avg_distance = lex_total_distance / len(lex_majority_selected)
        
        results['lexicographic'] = {
            'matched_data': lex_matched,
            'n_majority_selected': len(lex_majority_selected),
            'n_minority_selected': len(lex_minority_selected),
            'total_distance': lex_total_distance,
            'avg_distance_per_majority': lex_avg_distance,
            'computation_time': lex_time,
            'top_k_searched': top_k_majority,
            'success': True
        }
        
        if verbose:
            print(f"✓ Lexicographic completed in {lex_time:.2f}s")
            print(f"  Searched top {top_k_majority} majority candidates")
            print(f"  Selected: {len(lex_majority_selected)} majority, {len(lex_minority_selected)} minority")
            print(f"  Total distance: {lex_total_distance:.3f}")
            print(f"  Avg distance per majority: {lex_avg_distance:.3f}")
            
    except Exception as e:
        results['lexicographic'] = {'success': False, 'error': str(e)}
        if verbose:
            print(f"✗ Lexicographic failed: {e}")
    
    # ===== COMPARISON =====
    if results.get('ortools', {}).get('success') and results.get('lexicographic', {}).get('success'):
        ortools_dist = results['ortools']['avg_distance_per_majority']
        lex_dist = results['lexicographic']['avg_distance_per_majority']
        ortools_time = results['ortools']['computation_time']
        lex_time = results['lexicographic']['computation_time']
        
        distance_improvement = (ortools_dist - lex_dist) / ortools_dist * 100
        time_improvement = (ortools_time - lex_time) / ortools_time * 100
        
        results['comparison'] = {
            'distance_winner': 'lexicographic' if lex_dist < ortools_dist else 'ortools',
            'time_winner': 'lexicographic' if lex_time < ortools_time else 'ortools',
            'distance_improvement_pct': distance_improvement,
            'time_improvement_pct': time_improvement
        }
        
        if verbose:
            print(f"\n=== COMPARISON RESULTS ===")
            print(f"Distance quality:")
            print(f"  OR-Tools avg distance: {ortools_dist:.3f}")
            print(f"  Lexicographic avg distance: {lex_dist:.3f}")
            print(f"  Winner: {results['comparison']['distance_winner']}")
            print(f"  Improvement: {abs(distance_improvement):.1f}%")
            print(f"")
            print(f"Computation time:")
            print(f"  OR-Tools time: {ortools_time:.2f}s")
            print(f"  Lexicographic time: {lex_time:.2f}s") 
            print(f"  Winner: {results['comparison']['time_winner']}")
            print(f"  Speedup: {abs(time_improvement):.1f}%")
    
    return results


def run_comparison_on_smallest_leaves(pipeline_results, target_col, n_leaves=2, min_samples=50):
    """
    Run comparison on the N smallest leaves (by sample count)
    
    Parameters:
    -----------
    pipeline_results : dict
        Results from create_cost_strata_pipeline()
    target_col : str
        Target column for classification
    n_leaves : int
        Number of smallest leaves to test (default: 2)
    min_samples : int
        Minimum samples required for a leaf to be considered
        
    Returns:
    --------
    dict : Results for each tested leaf
    """
    
    # Initialize matcher instance
    matcher = EnhancedRiskBinnedCaseControlResampler(
    matching_method="ortools",  # Use OR-Tools for optimization
    random_state=42,
    binary_group="target_col",       # Your binary target column
    uid_col="ENROLID"            # Your ID column
    )
    
    # Get leaf data and find smallest leaves
    df_with_leaves = pipeline_results['df_with_strata_and_leaves']
    leaf_sizes = df_with_leaves['leaf_assignment'].value_counts().sort_values()
    
    print(f"=== SELECTING {n_leaves} SMALLEST LEAVES FOR COMPARISON ===")
    print(f"Total leaves: {len(leaf_sizes)}")
    print(f"Leaf size range: {leaf_sizes.min()} - {leaf_sizes.max()}")
    
    # Filter leaves with sufficient samples and class imbalance
    candidate_leaves = []
    for leaf_id, size in leaf_sizes.items():
        if size < min_samples:
            continue
            
        leaf_df = df_with_leaves[df_with_leaves['leaf_assignment'] == leaf_id]
        class_counts = leaf_df[target_col].value_counts()
        
        # Check if leaf has both classes and meaningful imbalance
        if len(class_counts) >= 2 and class_counts.min() >= 5:
            imbalance_ratio = class_counts.max() / class_counts.min()
            candidate_leaves.append((leaf_id, size, imbalance_ratio))
    
    # Sort by size and take smallest N
    candidate_leaves.sort(key=lambda x: x[1])  # Sort by size
    selected_leaves = candidate_leaves[:n_leaves]
    
    print(f"Selected leaves for comparison:")
    for leaf_id, size, imbalance in selected_leaves:
        print(f"  Leaf {leaf_id}: {size} samples, imbalance ratio {imbalance:.1f}:1")
    
    # Run comparison on selected leaves
    all_results = {}
    
    for leaf_id, size, imbalance in selected_leaves:
        print(f"\n{'='*60}")
        print(f"TESTING LEAF {leaf_id} (size: {size}, imbalance: {imbalance:.1f}:1)")
        print(f"{'='*60}")
        
        leaf_df = df_with_leaves[df_with_leaves['leaf_assignment'] == leaf_id].copy()
        
        try:
            results = compare_matching_methods_in_leaf(
                leaf_df=leaf_df,
                target_col=target_col,
                feature_cols=pipeline_results['stratification_cols'],
                matcher_instance=matcher,
                undersampling_ratio=1.0,
                top_k_factor=3.0,
                verbose=True
            )
            
            all_results[leaf_id] = results
            
        except Exception as e:
            print(f"✗ Error testing leaf {leaf_id}: {e}")
            all_results[leaf_id] = {'error': str(e)}
    
    # Summary across all tested leaves
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS {len(selected_leaves)} TESTED LEAVES")
    print(f"{'='*60}")
    
    successful_comparisons = []
    for leaf_id, results in all_results.items():
        if 'comparison' in results:
            comp = results['comparison']
            successful_comparisons.append({
                'leaf_id': leaf_id,
                'distance_winner': comp['distance_winner'],
                'time_winner': comp['time_winner'],
                'distance_improvement': comp['distance_improvement_pct'],
                'time_improvement': comp['time_improvement_pct'],
                'ortools_time': results['ortools']['computation_time'],
                'lex_time': results['lexicographic']['computation_time'],
                'ortools_distance': results['ortools']['avg_distance_per_majority'],
                'lex_distance': results['lexicographic']['avg_distance_per_majority']
            })
    
    if successful_comparisons:
        # Aggregate statistics
        distance_wins = {'lexicographic': 0, 'ortools': 0}
        time_wins = {'lexicographic': 0, 'ortools': 0}
        avg_distance_improvement = 0
        avg_time_improvement = 0
        
        for comp in successful_comparisons:
            distance_wins[comp['distance_winner']] += 1
            time_wins[comp['time_winner']] += 1
            avg_distance_improvement += comp['distance_improvement']
            avg_time_improvement += comp['time_improvement']
        
        n_success = len(successful_comparisons)
        avg_distance_improvement /= n_success
        avg_time_improvement /= n_success
        
        print(f"Successfully compared: {n_success}/{len(selected_leaves)} leaves")
        print(f"")
        print(f"Distance Quality:")
        print(f"  Lexicographic wins: {distance_wins['lexicographic']}/{n_success}")
        print(f"  OR-Tools wins: {distance_wins['ortools']}/{n_success}")
        print(f"  Average improvement: {abs(avg_distance_improvement):.1f}%")
        print(f"")
        print(f"Computation Speed:")
        print(f"  Lexicographic wins: {time_wins['lexicographic']}/{n_success}")
        print(f"  OR-Tools wins: {time_wins['ortools']}/{n_success}")
        print(f"  Average speedup: {abs(avg_time_improvement):.1f}%")
        
        # Show detailed results
        print(f"\nDetailed Results:")
        print(f"{'Leaf':<6} {'Size':<6} {'Distance Winner':<12} {'Time Winner':<12} {'OR-Tools Time':<12} {'Lex Time':<10}")
        print(f"{'-'*70}")
        for comp in successful_comparisons:
            print(f"{comp['leaf_id']:<6} "
                  f"{df_with_leaves[df_with_leaves['leaf_assignment']==comp['leaf_id']].shape[0]:<6} "
                  f"{comp['distance_winner']:<12} "
                  f"{comp['time_winner']:<12} "
                  f"{comp['ortools_time']:<12.2f} "
                  f"{comp['lex_time']:<10.2f}")
    
    return all_results