#!/bin/bash
# Example usage of xgboost_explainability.py

# Basic usage (parquet format)
python xgboost_explainability.py \
    --model_path trained_xgboost_model.pkl \
    --X_test_path X_test.parquet \
    --output_dir ./shap_results

# Basic usage (pickle format)
python xgboost_explainability.py \
    --model_path trained_xgboost_model.pkl \
    --X_test_path X_test.pkl \
    --output_dir ./shap_results

# With translation file (to use human-readable feature names)
# Note: If column_name_translations.csv exists in the current directory, 
#       it will be auto-detected even without --translation_file argument
python xgboost_explainability.py \
    --model_path trained_xgboost_model.pkl \
    --X_test_path X_test.pkl \
    --translation_file column_name_translations.csv \
    --output_dir ./shap_results

# Auto-detect translation file (if column_name_translations.csv is in current directory)
python xgboost_explainability.py \
    --model_path trained_xgboost_model.pkl \
    --X_test_path X_test.pkl \
    --output_dir ./shap_results

# With test labels (optional)
python xgboost_explainability.py \
    --model_path trained_xgboost_model.pkl \
    --X_test_path X_test.parquet \
    --y_test_path y_test.parquet \
    --output_dir ./shap_results

# With custom parameters (faster for large datasets)
python xgboost_explainability.py \
    --model_path trained_xgboost_model.pkl \
    --X_test_path X_test.parquet \
    --output_dir ./shap_results \
    --n_samples 1000 \
    --n_bootstrap 20 \
    --top_k 10 \
    --n_classes 4
