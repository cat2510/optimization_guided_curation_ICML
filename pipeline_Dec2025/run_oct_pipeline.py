#!/usr/bin/env python
"""
Main entry point for OCT pipeline with undersampling.

Usage:
    python run_oct_pipeline.py --config config/default_config.yaml
    python run_oct_pipeline.py --config config/default_config.yaml --method pushpull --ratio 1.0 --w 0.5
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path (pipeline_Dec2025 directory)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add parent directory to path (for balancing_functions and model_IAI)
parent_root = project_root.parent
sys.path.insert(0, str(parent_root))

from modules.utils import setup_logging, load_config, ensure_dir, get_exclude_cols_matching
from modules.data_loading import (
    load_data,
    split_features_labels,
    create_train_test_split,
    separate_cases_controls
)
from modules.preprocessing import detect_feature_types
from modules.sampling import apply_undersampling
from modules.train_oct import train_oct_with_tuning
from modules.evaluate_oct import evaluate_oct_model
from modules.update_metrics import update_metrics_master
from typing import List, Union, Any
import logging


def _run_single_ratio(
    final_ratio: float,
    X_train: Any,  # pd.DataFrame
    X_test: Any,  # pd.DataFrame
    y_train: Any,  # pd.Series
    y_test: Any,  # pd.Series
    feature_cols: List[str],
    categorical_cols: List[str],
    numeric_cols: List[str],
    exclude_cols_matching: List[str],
    data_config: dict,
    feature_config: dict,
    sampling_config: dict,
    oct_config: dict,
    output_config: dict,
    results_dir: str,
    logger: logging.Logger
) -> dict:
    """
    Run pipeline for a single ratio value.
    
    Returns
    -------
    dict
        Results dictionary for this ratio
    """
    # Combine train data for undersampling
    train_df = X_train.copy()
    train_df[data_config['target_col']] = y_train
    
    df_cases, df_controls = separate_cases_controls(
        df=train_df,
        target_col=data_config['target_col']
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running with ratio={final_ratio:.2f}")
    logger.info(f"{'='*60}")
    
    undersampled_df, ext, sampling_result = apply_undersampling(
            df_cases=df_cases,
            df_controls=df_controls,
            exclude_cols_matching=exclude_cols_matching,
            method=sampling_config['method'],
            final_ratio=final_ratio,
            w=sampling_config['w'],
            K_factor=sampling_config['K_factor'],
            top_k_case_ctrl=sampling_config['top_k_case_ctrl'],
            top_k_ctrl_ctrl=sampling_config.get('top_k_ctrl_ctrl', 20),
            L_pairs=sampling_config.get('L_pairs', 20),
            results_dir=results_dir,
            save_extreme_points_flag=sampling_config.get('save_extreme_points', True),
            load_extreme_points_flag=sampling_config.get('load_extreme_points', True),
            verbose=sampling_config.get('verbose', True),
            target_col=data_config['target_col'],  # Pass target_col to sampler
            uid_col=data_config['uid_col']  # Pass uid_col to sampler
        )
    
    # Prepare undersampled features and labels
    X_train_sampled = undersampled_df[feature_cols]
    y_train_sampled = undersampled_df[data_config['target_col']]
    
    # Train OCT
    logger.info(f"\nTraining OCT for ratio={final_ratio:.2f}...")
    X_train_oct = X_train_sampled
    y_train_oct = y_train_sampled
    X_val_oct = X_test
    y_val_oct = y_test
    
    best_model, best_params, tuning_results, preprocessor, feature_names = train_oct_with_tuning(
        X_train=X_train_oct,
        y_train=y_train_oct,
        X_val=X_val_oct,
        y_val=y_val_oct,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        depths=oct_config.get('depths', [5, 7, 9]),
        minbuckets=oct_config.get('minbuckets', [50, 100, 150]),
        cps=oct_config.get('cps', [0.001, 0.01, 0.05])
    )
    
    # Evaluate
    logger.info(f"Evaluating OCT for ratio={final_ratio:.2f}...")
    metrics = evaluate_oct_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        results_dir=results_dir if output_config.get('save_predictions', True) else None,
        method=sampling_config['method'],
        ratio=final_ratio,
        w=sampling_config['w'],
        compute_leaf_metrics=False
    )
    
    # Save metrics
    additional_fields = {
        'method': sampling_config['method'],
        'final_ratio': final_ratio,
        'w': sampling_config['w'],
        'best_depth': best_params[0],
        'best_minbucket': best_params[1],
        'best_cp': best_params[2],
        'f1': sampling_result.get('f1', None),
        'f2': sampling_result.get('f2', None),
    }
    
    update_metrics_master(
        metrics=metrics,
        results_dir=results_dir,
        method=sampling_config['method'],
        metrics_file=output_config.get('metrics_file'),
        additional_fields=additional_fields
    )
    
    return {
        'model': best_model,
        'metrics': metrics,
        'sampling_result': sampling_result,
        'extreme_points': ext,
        'ratio': final_ratio
    }


def run_pipeline(config_path: str, **overrides):
    """
    Run the complete OCT pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file
    **overrides : dict
        Configuration overrides (e.g., method="pushpull", ratio=1.0)
        
    Returns
    -------
    dict
        Results dictionary with model, metrics, sampling_result, extreme_points, log_file
    """
    tee_output = None
    log_file = None
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Apply overrides
        for key, value in overrides.items():
            if value is not None:
                # Handle nested keys (e.g., "undersampling.method")
                keys = key.split('.')
                d = config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
        
        # Extract configuration
        data_config = config['data']
        feature_config = config.get('features', {})
        sampling_config = config['undersampling']
        oct_config = config['oct']
        output_config = config['output']
        
        results_dir = output_config['results_dir']
        ensure_dir(results_dir)
        
        # Check if final_ratio is a list (for ratio sweeps)
        final_ratio_raw = sampling_config.get('final_ratio', 1.0)
        if isinstance(final_ratio_raw, list):
            final_ratios = final_ratio_raw
            is_ratio_sweep = True
        else:
            final_ratios = [final_ratio_raw]
            is_ratio_sweep = False
        
        # Setup logging with file output
        logging_config = config.get('logging', {})
        log_level = logging_config.get('level', 'INFO')
        log_to_file = logging_config.get('log_to_file', True)
        capture_print = logging_config.get('capture_print', True)
        
        # Generate log file path
        if log_to_file:
            log_dir = os.path.join(results_dir, 'logs')
            ensure_dir(log_dir)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            method = sampling_config.get('method', 'unknown')
            w = sampling_config.get('w', 0.5)
            if is_ratio_sweep:
                ratio_str = f"ratios_{min(final_ratios):.0f}to{max(final_ratios):.0f}"
            else:
                ratio_str = f"ratio_{final_ratios[0]:.2f}"
            log_filename = f"pipeline_{method}_{ratio_str}_w_{w:.2f}_{timestamp}.log"
            log_file = os.path.join(log_dir, log_filename)
        
        logger, tee_output = setup_logging(
            level=log_level,
            log_file=log_file,
            capture_print=capture_print
        )
        
        if log_file:
            logger.info(f"Logging to file: {log_file}")
        
        logger.info("="*60)
        logger.info("Starting OCT Pipeline")
        if is_ratio_sweep:
            logger.info(f"Ratio sweep: {final_ratios}")
        logger.info("="*60)
        
        # ====================================================================
        # Step 1: Load and split data
        # ====================================================================
        logger.info("\n" + "="*60)
        logger.info("Step 1: Loading and splitting data")
        logger.info("="*60)
    
        df = load_data(
            data_path=data_config['data_path'],
            uid_col=data_config['uid_col'],
            file_format=data_config.get('file_format', 'auto')
        )
        
        # Build exclude_cols_matching with COST_COLUMNS and cost_stratum_2018
        exclude_cols_matching = get_exclude_cols_matching(
            df=df,
            exclude_cols_matching=data_config.get('exclude_cols_matching', []),
            include_cost_columns=True,
            include_cost_stratum=True
        )
        if exclude_cols_matching:
            logger.info(f"Excluding columns from matching: {exclude_cols_matching}")
        
        X, y, feature_cols = split_features_labels(
            df=df,
            target_col=data_config['target_col'],
            uid_col=data_config['uid_col'],
            exclude_cols=exclude_cols_matching,
            exclude_cost_cols=data_config.get('exclude_cost_cols', True),
            exclude_high_corr=data_config.get('exclude_high_corr', True),
            correlation_threshold=data_config.get('correlation_threshold', 0.5)
        )
        
        X_train, X_test, y_train, y_test = create_train_test_split(
            X=X,
            y=y,
            df_original=df,  # Pass original df for patient-level splitting
            test_size=data_config.get('test_size', 0.2),
            random_state=data_config.get('random_state', 42),
            stratify=data_config.get('stratify', True),
            patient_level=data_config.get('patient_level_split', False),
            patient_id_col=data_config.get('patient_id_col', 'ENROLID')
        )
        
        # ====================================================================
        # Step 2: Detect feature types
        # ====================================================================
        logger.info("\n" + "="*60)
        logger.info("Step 2: Detecting feature types")
        logger.info("="*60)
        
        categorical_cols, numeric_cols = detect_feature_types(
            df=X_train,
            target_col=data_config['target_col'],
            uid_col=data_config['uid_col'],
            exclude_cols=exclude_cols_matching,
            use_model_pipeline_categories=feature_config.get('use_model_pipeline_categories', True)
        )
        
        # Override with config if provided
        if feature_config.get('categorical_cols'):
            categorical_cols = feature_config['categorical_cols']
        if feature_config.get('numeric_cols'):
            numeric_cols = feature_config['numeric_cols']
        
        # ====================================================================
        # Step 3-6: Run pipeline for each ratio
        # ====================================================================
        all_results = []
        
        for ratio_idx, final_ratio in enumerate(final_ratios):
            if is_ratio_sweep:
                logger.info(f"\n{'='*60}")
                logger.info(f"Ratio {ratio_idx + 1}/{len(final_ratios)}: {final_ratio:.2f}")
                logger.info(f"{'='*60}")
            
            result = _run_single_ratio(
                final_ratio=final_ratio,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                exclude_cols_matching=exclude_cols_matching,
                data_config=data_config,
                feature_config=feature_config,
                sampling_config=sampling_config,
                oct_config=oct_config,
                output_config=output_config,
                results_dir=results_dir,
                logger=logger
            )
            all_results.append(result)
        
        logger.info("\n" + "="*60)
        logger.info("Pipeline completed successfully!")
        if is_ratio_sweep:
            logger.info(f"Completed ratio sweep: {final_ratios}")
        logger.info("="*60)
        
        # Restore stdout if we redirected it
        if tee_output:
            sys.stdout = tee_output.terminal
            tee_output.close()
            logger.info(f"Log file saved to: {log_file}")
        
        # Return last result (or all results if sweep)
        if is_ratio_sweep:
            return {
                'all_results': all_results,
                'ratios': final_ratios,
                'log_file': log_file
            }
        else:
            return {
                'model': all_results[0]['model'],
                'metrics': all_results[0]['metrics'],
                'sampling_result': all_results[0]['sampling_result'],
                'extreme_points': all_results[0]['extreme_points'],
                'log_file': log_file
            }
    
    except Exception as e:
        # Restore stdout if we redirected it (even on error)
        if tee_output:
            sys.stdout = tee_output.terminal
            tee_output.close()
        # Re-raise the exception
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run OCT pipeline with undersampling',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration YAML file'
    )
    
    # Allow overriding key parameters
    parser.add_argument('--method', type=str, choices=['pushpull', 'doublefacility'],
                       help='Undersampling method')
    parser.add_argument('--ratio', type=float, help='Final ratio (controls:cases)')
    parser.add_argument('--w', type=float, help='Weight for multi-objective optimization')
    parser.add_argument('--data-path', type=str, help='Path to data file')
    
    args = parser.parse_args()
    
    # Prepare overrides
    overrides = {}
    if args.method:
        overrides['undersampling.method'] = args.method
    if args.ratio is not None:
        overrides['undersampling.final_ratio'] = args.ratio
    if args.w is not None:
        overrides['undersampling.w'] = args.w
    if args.data_path:
        overrides['data.data_path'] = args.data_path
    
    # Run pipeline
    try:
        results = run_pipeline(args.config, **overrides)
        print("\n✓ Pipeline completed successfully!")
        if results.get('log_file'):
            print(f"✓ Log file: {results['log_file']}")
        return 0
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        # Make sure to restore stdout if it was redirected
        if hasattr(sys.stdout, 'terminal'):
            sys.stdout = sys.stdout.terminal
        return 1


if __name__ == '__main__':
    sys.exit(main())

