# ML Pipeline for High-Cost Prediction

This pipeline allows systematic testing of different `highcost_cutoff` values for predicting high-cost patients in your CKD cohort, including matching and balancing procedures.

## Quick Start

### Method 1: Simple Command Line Execution

```bash
# Make the script executable
chmod +x run_pipeline.sh

# Run with default settings
./run_pipeline.sh

# Run with specific cutoffs and models
./run_pipeline.sh -k 3,4,5 -m logistic,random_forest

# Run with custom config
./run_pipeline.sh -c my_config.json
```

### Method 2: Direct Python Execution

```bash
# Activate your conda environment
conda activate analytics

# Run with default config
python ml_pipeline.py

# Run with command line arguments
python ml_pipeline.py --highcost-cutoffs 3 4 5 --models logistic random_forest
```

### Method 3: Batch Experiments (Multiple Configurations)

```bash
# Run multiple parameter combinations
python batch_runner.py --mode run

# Combine results from all experiments
python batch_runner.py --mode combine

# Run experiments and combine results
python batch_runner.py --mode both
```

## Configuration

### Basic Configuration File (config.json)

```json
{
  "data_path": "0813_cost_features_2017.parquet",
  "highcost_cutoffs": [3, 4, 5],
  "models": ["logistic", "random_forest", "gradient_boosting"],
  "test_size": 0.3,
  "output_dir": "pipeline_output"
}
```

### Available Models

- `logistic`: Logistic Regression
- `random_forest`: Random Forest
- `gradient_boosting`: Gradient Boosting (HistGradientBoosting)
- `oct`: Optimal Classification Trees (IAI)
- `opt`: Optimal Policy Trees (IAI)

### Key Parameters

- `highcost_cutoffs`: List of cost stratum cutoffs to test (e.g., [3, 4, 5])
- `stratifier_cutoff`: Cutoff for stratification matching (default: 3)
- `test_size`: Fraction of data for testing (default: 0.3)
- `balance_classes`: Whether to balance classes in gradient boosting
- `matching.enabled`: Whether to perform propensity score matching

## File Structure

```
your_project/
├── ml_pipeline.py          # Main pipeline script
├── batch_runner.py         # Batch experiment runner
├── analyze_results.py      # Results analysis script
├── run_pipeline.sh         # Shell script for easy execution
├── config.json            # Configuration file
├── pipeline_output/        # Default output directory
├── batch_output/          # Batch experiment outputs
└── README.md              # This file
```

## Usage Examples

### Example 1: Test Different Cutoffs

```bash
# Test cutoffs 3, 4, 5, 6 with all models
./run_pipeline.sh -k 3,4,5,6 -m logistic,random_forest,gradient_boosting
```

### Example 2: Focus on Best Model

```bash
# Run only gradient boosting with different cutoffs
./run_pipeline.sh -k 2,3,4,5,6 -m gradient_boosting
```

### Example 3: Custom Configuration

Create `high_cutoff_config.json`:
```json
{
  "data_path": "0813_cost_features_2017.parquet",
  "highcost_cutoffs": [4, 5, 6],
  "models": ["gradient_boosting"],
  "balance_classes": true,
  "test_size": 0.25,
  "output_dir": "high_cutoff_results"
}
```

Run with:
```bash
./run_pipeline.sh -c high_cutoff_config.json
```

## Analyzing Results

### Method 1: Automated Analysis

```bash
# Analyze results from a specific run
python analyze_results.py pipeline_output/pipeline_results_20241201_143022.csv

# Analyze with custom output directory
python analyze_results.py results.csv --output-dir analysis_results
```

### Method 2: Manual Analysis

The pipeline outputs CSV files with the following key columns:
- `Model`: Model name
- `Dataset`: Before/After Matching
- `highcost_cutoff`: The cutoff value used
- `AUC`, `Precision`, `Recall`, `F1`: Performance metrics
- `Risk_Bin`: Performance by risk stratification

## Output Files

### Single Run Output
- `pipeline_results_TIMESTAMP.csv`: Detailed results
- `config_TIMESTAMP.json`: Configuration used
- `pipeline.log`: Execution log

### Batch Run Output
- `batch_output/`: Directory containing all experiments
- `combined_results.csv`: Combined results from all experiments
- `batch_results_summary.csv`: Success/failure summary
- `combined_analysis/`: Automated analysis plots and reports

### Analysis Output
- `*_heatmap.png`: Performance heatmaps
- `performance_trends.png`: Performance trends across cutoffs
- `optimal_cutoffs_*.csv`: Best cutoffs for each metric
- `analysis_report.md`: Comprehensive markdown report

## Integration with Your Notebook

To integrate with your existing notebook workflow:

1. **Extract your data preparation code** from the notebook into the pipeline
2. **Add matching logic** in the `run_single_experiment` method
3. **Customize evaluation metrics** in the `evaluate_models` method

### Adding Custom Models

Add to the `train_models` method in `ml_pipeline.py`:

```python
# Your custom model
if 'my_model' in self.config.get('models', []):
    logger.info("Training My Custom Model...")
    models['my_model'] = your_model_pipeline_function(
        X_train, y_train, **your_params
    )
```

## Troubleshooting

### Common Issues

1. **Data file not found**: Check the `data_path` in your config
2. **Module import errors**: Ensure all your custom modules are in the Python path
3. **Memory issues**: Try reducing batch size or using fewer models
4. **IAI models failing**: These require special licenses - they'll be skipped if unavailable

### Debug Mode

Add `"verbose": true` to your config for detailed logging.

### Checking Results

```bash
# Check if pipeline completed successfully
echo $?  # Should be 0 for success

# View recent log
tail -f pipeline.log

# Check output directory
ls -la pipeline_output/
```

## Extending the Pipeline

### Adding Matching Logic

Modify the `run_single_experiment` method to include your matching procedure:

```python
# Add after model training
if self.config.get('matching', {}).get('enabled', True):
    # Your matching logic here
    matched_train_df, matched_test_df = perform_matching(
        train_df, test_df, method='propensity_score'
    )
    
    # Re-evaluate on matched data
    results_after = self.evaluate_models(
        models, matched_X_test, matched_y_test, 
        matched_test_df, highcost_cutoff, "after"
    )
```

### Adding New Evaluation Metrics

Extend the evaluation functions to include your custom metrics from the notebook.

## Performance Optimization

- Use `save_models: false` to reduce disk usage
- Set `cross_validation.enabled: false` for faster runs  
- Use fewer models for initial exploration
- Consider parallel execution for batch runs

## Support

For issues with the pipeline, check:
1. Pipeline logs (`pipeline.log`)
2. Individual experiment logs in output directories
3. Ensure all required modules are available
4. Verify data file paths and formats

The pipeline is designed to be fault-tolerant - if one cutoff fails, others will continue running.
