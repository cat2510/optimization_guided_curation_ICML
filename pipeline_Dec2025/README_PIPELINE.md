# OCT Pipeline with Undersampling

Modular pipeline for training Optimal Classification Trees (OCT) with global undersampling (pushpull or doublefacility).

## Directory Structure

```
pipeline_Dec2025/
│
├── run_oct_pipeline.py              # Main entry point
│
├── config/
│   └── default_config.yaml          # Configuration file
│
├── modules/
│   ├── __init__.py
│   ├── data_loading.py             # Data loading and splitting
│   ├── preprocessing.py             # Feature type detection
│   ├── sampling.py                  # Undersampling with extreme point caching
│   ├── train_oct.py                 # OCT training with hyperparameter tuning
│   ├── evaluate_oct.py              # Model evaluation
│   ├── save_predictions.py         # Save predictions
│   ├── update_metrics.py           # Update metrics master file
│   └── utils.py                     # Utility functions
│
└── results/
    ├── metrics_master.csv           # Aggregated metrics
    ├── extreme_points/              # Cached extreme points (JSON)
    ├── predictions/                 # OCT predictions (CSV)
    └── models/                      # Saved models (optional)
```

**Note**: This pipeline is located in `pipeline_Dec2025/` and imports from:
- `balancing_functions/` (parent directory)
- `model_IAI.py` (parent directory)

## Usage

### Basic Usage

From the `pipeline_Dec2025/` directory:

```bash
cd pipeline_Dec2025
python run_oct_pipeline.py --config config/default_config.yaml
```

Or from the project root:

```bash
python pipeline_Dec2025/run_oct_pipeline.py --config pipeline_Dec2025/config/default_config.yaml
```

### With Command-Line Overrides

```bash
cd pipeline_Dec2025
python run_oct_pipeline.py \
    --config config/default_config.yaml \
    --method pushpull \
    --ratio 1.0 \
    --w 0.5 \
    --data-path path/to/your/data.csv
```

### Ratio Sweeps

The pipeline supports running multiple ratios in a single execution (similar to `ratio_sweep_doublefacility.ipynb`):

```yaml
undersampling:
  method: "pushpull"
  final_ratio: [1, 2, 3, 4, 5]  # List of ratios to sweep
  w: 0.5
```

When `final_ratio` is a list, the pipeline will:
- Run the complete pipeline for each ratio
- Save separate predictions and metrics for each ratio
- Append all metrics to the same `metrics_master_{method}.csv` file
- Create a single log file for the entire sweep

### Configuration File

Edit `config/default_config.yaml` to set:
- Data paths and column names
- Undersampling parameters (method, ratio, weights)
  - `final_ratio` can be a single value (e.g., `1.0`) or a list for sweeps (e.g., `[1, 2, 3, 4, 5]`)
- OCT hyperparameter search space
- Output directories

## Pipeline Steps

1. **Data Loading**: Load dataset and split into features/labels
2. **Train-Test Split**: Create outcome-stratified train-test split
3. **Feature Detection**: Automatically detect categorical and numeric columns
4. **Undersampling**: Apply pushpull or doublefacility undersampling
   - Computes or loads cached extreme points
   - Saves undersampled training data
5. **OCT Training**: Train OCT with hyperparameter tuning
6. **Evaluation**: Evaluate on test set and compute metrics
7. **Save Results**: Save predictions and update metrics master file

## Extreme Point Caching

The pipeline automatically caches extreme points to avoid recomputing them:

- **Save**: Extreme points are saved to `results/extreme_points/` as JSON files
- **Load**: On subsequent runs with same parameters, cached extreme points are loaded
- **Cache Key**: Based on method, ratio, and pruning parameters

## Output Files

All output files are method-specific to avoid overwriting results from different undersampling methods:

- `results/metrics_master_{method}.csv`: Aggregated metrics for each method (e.g., `metrics_master_pushpull.csv`, `metrics_master_doublefacility.csv`)
- `results/predictions/{method}/oct_predictions_{method}_ratio_{ratio:.2f}_w_{w:.2f}.csv`: Test set predictions (organized by method)
- `results/extreme_points/extreme_points_{method}_*.json`: Cached extreme points (already method-specific)
- `results/undersampled_{method}_ratio_{ratio:.2f}_w_{w:.2f}.csv`: Undersampled training data (already method-specific)

## Example Workflow

```python
# 1. Configure in config/default_config.yaml
# 2. Run pipeline
python run_oct_pipeline.py --config config/default_config.yaml

# 3. Check results
cat results/metrics_master_pushpull.csv  # or metrics_master_doublefacility.csv
ls results/predictions/pushpull/  # or predictions/doublefacility/
```

## Logging

The pipeline automatically logs all output to a file for debugging:

- **Log location**: `results/logs/pipeline_{method}_ratio_{ratio:.2f}_w_{w:.2f}_{timestamp}.log`
- **Captures**: All logging messages AND print() statements
- **Configurable**: Set `logging.log_to_file` and `logging.capture_print` in config
- **Format**: Timestamped log entries with level, module, and message

Example log file:
```
2025-01-15 10:30:45 - __main__ - INFO - Starting OCT Pipeline
2025-01-15 10:30:45 - __main__ - INFO - Loading data from: data.csv
2025-01-15 10:30:46 - modules.sampling - INFO - Computing extreme points...
✓ Saved extreme points to: results/extreme_points/...
```

## Notes

- Extreme points are expensive to compute (4 MILP solves), so caching is highly recommended
- The pipeline uses the test set as validation for hyperparameter tuning (consider splitting undersampled train data for true validation)
- All paths are relative to the `pipeline_Dec2025/` directory when running from that location
- The pipeline automatically adds the parent directory to `sys.path` to import `balancing_functions` and `model_IAI`
- Log files are automatically created in `results/logs/` with timestamps for easy debugging

