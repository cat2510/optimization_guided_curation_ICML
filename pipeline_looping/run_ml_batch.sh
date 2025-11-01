#!/bin/bash

# Background ML Pipeline Runner
# This script runs 20 ML experiments with different random seeds
# Designed to be run inside a screen session

echo "ML Pipeline Batch Runner - 20 experiments with different seeds"
echo "Started at: $(date)"
echo "Process ID: $$"
echo ""

# Create log file with timestamp
LOG_FILE="ml_pipeline_batch_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

# Function to log and print
log_and_print() {
    echo "$1" | tee -a "$LOG_FILE"
}
# Function to create output directory if it doesn't exist
create_output_dir() {
    local seed=$1
    local output_dir="output_0911_stratifier_high_stage_seed${seed}"
    
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
        log_and_print "Created output directory: $output_dir"
    fi
}
# Function to create config file for specific seed
create_config() {
    local seed=$1
    local config_file="config_seed_${seed}.json"
    
    cat > "$config_file" << EOF
{
  "data_path": "../0902_adherence_income_info.parquet",
  "cutoff_colnames":["highcost_gt_50000", "highcost_gt_75000", 
        "highcost_gt_100000", "highcost_gt_200000","highcost_gt_300000",
        "highcost_gt_400000","highcost_gt_500000"],
  "stratifier_cutoff": 3,
  "stratifier_target_column": "annual_cost17",
  "test_size": 0.3,
  "models": [
    "logistic",
    "random_forest", 
    "gradient_boosting","oct"
  ],
  "logistic_C": 1,
  "logistic_max_iter": 1000,
  "balance_classes": true,
  "output_dir": "output_0911_stratifier_high_stage_seed${seed}",
  "random_seed": $seed,
  "apply_matching": true,
  "matching_methruod": "ortools",
  "feature_selection": {
    "correlation_threshold": 0.5,
    "remove_highly_correlated": true
  },
  "evaluation_metrics": [
    "auc",
    "precision",
    "recall",
    "f1"
  ],
  "save_models": false,
  "save_plots": true,
  "verbose": true
}
EOF
    echo "$config_file"
}

# Array of 20 different random seeds for confidence intervals
seeds=(1 123 456 789 1011 1213 1415 1617 1819 2021 
       2223 2425 2627 2829 3031 3233 3435 3637 3839 4041)

# Track successful and failed runs
successful_runs=0
failed_runs=0
failed_seeds=()

log_and_print "========================================="
log_and_print "Starting 20 ML pipeline runs for confidence intervals"
log_and_print "Seeds: ${seeds[*]}"
log_and_print "========================================="

# Main execution loop
for i in {1..20}; do
    seed=${seeds[$((i-1))]}
    
    log_and_print ""
    log_and_print "Run $i/20 - Seed: $seed - $(date)"
    log_and_print "-----------------------------------------"
    
    # Create config file
    config_file=$(create_config $seed)
     # Create output directory
    create_output_dir $seed
    # Run the pipeline
    log_and_print "Executing: python ml_pipeline.py --config $config_file"
    
    # Run and capture both stdout/stderr
    python ml_pipeline.py --config "$config_file" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    
    # Check success
    if [ $exit_code -eq 0 ]; then
        log_and_print "✓ Run $i completed successfully (seed: $seed)"
        ((successful_runs++))
    else
        log_and_print "✗ Run $i FAILED (seed: $seed, exit code: $exit_code)"
        failed_seeds+=($seed)
        ((failed_runs++))
    fi
    
    # Clean up config file
    rm -f "$config_file"
    
    log_and_print "Completed run $i at $(date)"
    
    # Progress update
    remaining=$((20 - i))
    log_and_print "Progress: $i/20 complete, $remaining remaining"
done

# Final summary
log_and_print ""
log_and_print "========================================="
log_and_print "BATCH COMPLETION SUMMARY"
log_and_print "========================================="
log_and_print "Total runs: 20"
log_and_print "Successful: $successful_runs"
log_and_print "Failed: $failed_runs"
log_and_print "Success rate: $(( successful_runs * 100 / 20 ))%"

if [ ${#failed_seeds[@]} -gt 0 ]; then
    log_and_print "Failed seeds: ${failed_seeds[*]}"
fi

log_and_print "Started: $(date)"
log_and_print "Completed: $(date)"
log_and_print "Log file: $LOG_FILE"
log_and_print "========================================="

echo ""
echo "All experiments completed!"
echo "Check $LOG_FILE for detailed logs"
echo "Results saved in separate directories for each seed"
echo ""
echo "You can now:"
echo "1. Analyze confidence intervals across the 20 runs"
echo "2. Aggregate results from output_*_seed* directories"
echo "3. Exit this screen session (Ctrl-A D to detach, or 'exit' to close)"