#!/usr/bin/env python3
"""
Machine Learning Pipeline for High-Cost Prediction
Supports multiple highcost_cutoff values and matching strategies
"""

import sys

print(f"Python version: {sys.version}")
import json
import logging
import pandas as pd
import numpy as np
from functools import reduce
from pyspark.sql.types import StringType,DecimalType,DoubleType,IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler, Bucketizer
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession,Row
from pyspark import SparkConf
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import argparse
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to Python path to import model_pipeline
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import your existing modules
import model_pipeline
from model_IAI import train_oct_with_feature_names, train_opt_with_feature_names
from evaluate_model_by_risk_bin import evaluate_metrics_by_risk_bin, plot_probability_and_calibration

# -----------------------------------------------------------------------------
# INITIALIZE LOGGING
# -----------------------------------------------------------------------------
f = '%(asctime)-15s %(levelname)-8s %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logging.basicConfig(format=f)
try:
    from balancing_functions.optimal_match_control import EnhancedRiskBinnedCaseControlResampler, get_high_cost_detection_config
    #from balancing_functions.optimal_match_control_jaccard import EnhancedRiskBinnedCaseControlResampler, get_high_cost_detection_config

    MATCHING_AVAILABLE = True
except ImportError:
    MATCHING_AVAILABLE = False
    logger.warning("Matching modules not available")
# -----------------------------------------------------------------------------
# start_spark
# -----------------------------------------------------------------------------
def start_spark(
    driver_memory="100g",
    storage_fraction=0.5,
    num_nodes=10,
):
    """Initialize spark"""
    conf = SparkConf().setAppName("My_Application")
    conf.set("spark.driver.memory", driver_memory)
    conf.set("spark.memory.storageFraction", str(storage_fraction))
    conf.setMaster(f"local[{num_nodes}]")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    return spark

# Initialize Spark session
spark = start_spark(num_nodes=10)

class MLPipeline:
    def __init__(self, config):
        """
        Initialize the ML Pipeline
        
        Args:
            config (dict): Configuration dictionary containing parameters
        """
        self.config = config
        self.results = []
        self.models = {}
        self.feature_cols = []
        # Define cost cutoffs and column names
        self.cutoffs = [50000, 75000, 100000, 200000, 300000, 400000, 500000]
        self.cutoff_colnames = [f"highcost_gt_{c}" for c in self.cutoffs]
        
        # Stratification hyperparameters
        self.stratifier_stage_cutoff = config.get('stratifier_stage_cutoff', 3)
        self.stratifier_model_type = config.get('stratifier_model_type', 'xgb_multiclass')
        self.n_strata = config.get('n_strata', 8)

        # Feature column definitions (from your notebook)
        self.BIN_FLAG_COLUMNS = None
        self.STAGE_COLUMNS = [#'2017Q1', '2017Q2', '2017Q3', '2017Q4', 
                             "stage_2017",'2017Q1_max_ckd_stage', '2017Q2_max_ckd_stage', 
                             '2017Q3_max_ckd_stage', '2017Q4_max_ckd_stage']
        self.CAT_COLUMNS = ['ENROLID', 'AGEGRP', 'SEX', 'REGION', 'INDSTRY']
        self.TRUE_NUM_COLUMNS = None
        
    def get_output_dir(self):
        """Get consistent output directory"""
        base_dir = Path(self.config.get('output_dir', 'pipeline_output'))
        return base_dir
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset using Spark"""
        logger.info("Loading and preparing data with Spark...")
        
        # Load data using Spark
        data_path = self.config.get('data_path', '0827_2017_18_highcost_cutoffs.parquet')
        
        if data_path.endswith('.parquet'):
            df_spark = spark.read.format("parquet").load(data_path)
            self.df_og = df_spark.toPandas()
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Fill missing values
        if "INDSTRY" in self.df_og.columns:
            self.df_og["INDSTRY"] = self.df_og["INDSTRY"].fillna("-1")        
        # Define feature columns
        self.BIN_FLAG_COLUMNS = model_pipeline.get_bin_flag_columns(self.df_og) + [
            'lab_monitoring_adherent', 'nephrology_consult_adherent', 'early_nephrology_referral'
        ]
        
        self.CAT_COLUMNS = self.df_og.select_dtypes(include=["object", "category"]).columns.tolist()
        self.TRUE_NUM_COLUMNS = model_pipeline.get_true_num_columns(
            self.df_og, self.CAT_COLUMNS
        ) + ['util_2017', 'total_increasing_quarters_2017', 'total_lab_tests',"MEDIAN_INCOME"]
        
        logger.info(f"Data loaded with Spark. Shape: {self.df_og.shape}")
        
    def prepare_features(self):
        """Prepare feature columns independent of any specific target"""
        logger.info("Preparing feature columns...")
        
        # Get feature columns (excluding ALL cutoff columns and metadata)
        cutoff_columns = [col for col in self.df_og.columns if col.startswith('highcost_gt_')]
        exclude_cols = ["ENROLID", "annual_cost17", "cost_stratum_2018"] + cutoff_columns
        
        feature_cols = [c for c in self.df_og.columns if c not in exclude_cols]
        
        # Remove highly correlated features using annual_cost17 as reference
        numeric_cols = self.df_og[feature_cols + ["annual_cost17"]].select_dtypes(
            include=["number"]
        ).columns
        corrs = self.df_og[numeric_cols].corr()["annual_cost17"].abs().sort_values(ascending=False)
        
        high_corr_cols = corrs[corrs > 0.5].index.tolist()
        high_corr_cols = [col for col in high_corr_cols if col != "annual_cost17"]
        
        feature_cols = [col for col in feature_cols if col not in high_corr_cols]
    
        logger.info(f"Selected {len(feature_cols)} features, removed {len(high_corr_cols)} highly correlated")
        
        return feature_cols
    
    def create_risk_bins(self):
        logger.info(f"Creating risk bins using stage cutoff > {self.stratifier_stage_cutoff}")
        # Train stratifier on configurable stage cutoff
        stratifier_train_pd = self.df_og[self.df_og["stage_2017"] > self.stratifier_stage_cutoff].copy()
        logger.info(f"Stratifier trained on subgroup of size {len(stratifier_train_pd)}")
        
        # Remove stage columns from stratification features
        stratification_cols = [c for c in self.feature_cols if c not in self.STAGE_COLUMNS]
       
        logger.info(f"Using '{self.stratifier_model_type}' as stratification method")
        # Prepare target
        if self.stratifier_model_type == 'regression':
            stratifier_ytrain = self.df_og.loc[stratifier_train_pd.index, "annual_cost17"]
        elif self.stratifier_model_type == 'xgb_multiclass':
            # Create cost strata with quantile binning
            stratifier_train_pd['cost_stratum'] = pd.qcut(
                stratifier_train_pd["annual_cost17"], 
                q= self.n_strata, 
                labels=list(range(self.n_strata)),
                duplicates='drop'  # Handle ties
            )
            stratifier_ytrain = stratifier_train_pd['cost_stratum']
        
        # Train stratifier
        stratifier, risk_converter = model_pipeline.get_stratifier_model(
            df=stratifier_train_pd[stratification_cols],
            categorical_cols=self.CAT_COLUMNS, 
            numeric_cols=self.TRUE_NUM_COLUMNS,
            model_type=self.stratifier_model_type)
        stratifier.fit(stratifier_train_pd[stratification_cols], stratifier_ytrain)
        
        df_binned = self.df_og.copy()
        if self.stratifier_model_type == 'regression':
            predictions = stratifier.predict(df_binned[stratification_cols])
            training_predictions = stratifier.predict(stratifier_train_pd[stratification_cols])  # For ECDF reference
            df_binned["risk_score"] = risk_converter(predictions, training_predictions)
            # Create risk bins using fixed probability bins
            bin_edges = np.linspace(0, 1, 11)
            bin_labels = [f"{a:.2f}-{b:.2f}" for a, b in zip(bin_edges[:-1], bin_edges[1:])]
            
            df_binned["risk_bin"] = pd.cut(
                df_binned["risk_score"],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True,
                right=True,
            )
        elif self.stratifier_model_type == 'xgb_multiclass':
            df_binned["risk_bin"] = stratifier.predict(df_binned[stratification_cols])
            bin_counts = df_binned["risk_bin"].value_counts().sort_index()
            logger.info("Sample counts per risk_bin:\n%s", bin_counts.to_string())
        return df_binned
        
    def plot_risk_distribution(self, df_binned, matched=False, risk_bin_type='auto'):
        """
        Plot risk distribution by stage group for training set
        
        Args:
            df_binned: DataFrame with risk_bin and stage columns
            matched: Whether this is a matched/balanced dataset
            risk_bin_type: 'auto', 'integer', 'range', or 'detect'
                - 'auto'/'detect': Automatically detect format
                - 'integer': risk_bin contains integers like "0", "1", "2"
                - 'range': risk_bin contains ranges like "0.00-0.10"
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            logger.info(f"Train size: {len(df_binned)}")
            
            # Create true_class column for stage grouping
            df = df_binned.copy()
            if "true_class" not in df.columns:
                df["true_class"] = df["stage_2017"].apply(
                    lambda x: f"Stage ≤{self.stratifier_stage_cutoff}" if x <= self.stratifier_stage_cutoff else f"Stage >{self.stratifier_stage_cutoff}"
                )
            
            # Auto-detect risk_bin format if needed
            if risk_bin_type in ['auto', 'detect']:
                sample_bin = str(df['risk_bin'].iloc[0])
                if '-' in sample_bin and '.' in sample_bin:
                    risk_bin_type = 'range'
                else:
                    risk_bin_type = 'integer'
                logger.info(f"Auto-detected risk_bin_type: {risk_bin_type}")
            
            # Handle sorting and labeling based on type
            if risk_bin_type == 'integer':
                risk_bins = df['risk_bin'].unique()
                try:
                    risk_bins_sorted = sorted(risk_bins, key=lambda x: int(x))
                    x_labels = [f"Stratum {bin_val}" for bin_val in risk_bins_sorted]
                    xlabel = "Predicted Cost Stratum"
                except ValueError:
                    # Fallback if conversion fails
                    risk_bins_sorted = sorted(risk_bins)
                    x_labels = risk_bins_sorted
                    xlabel = "Predicted Risk Bin"
                    
            elif risk_bin_type == 'range':
                # Range-based bins: "0.00-0.10", "0.10-0.20", etc.
                risk_bins = df['risk_bin'].unique()
                risk_bins_sorted = sorted(risk_bins, key=lambda x: float(x.split('-')[0]))
                x_labels = risk_bins_sorted
                xlabel = "Predicted Risk Score"
            
            else:
                raise ValueError(f"Unknown risk_bin_type: {risk_bin_type}")
            
            # Create matplotlib plot
            plt.figure(figsize=(12, 6))
            
            stage_groups = df['true_class'].unique()
            x = np.arange(len(risk_bins_sorted))
            width = 0.35
            
            if matched:
                plot_title = "Balanced Train Set: Patient Distribution by Risk Stratum and CKD Stage Group"
            else:
                plot_title = "Original Train Set: Patient Distribution by Risk Stratum and CKD Stage Group"

            # Create grouped bar plot
            colors = ['lightblue', 'coral']
            for i, stage_group in enumerate(stage_groups):
                counts = df[df['true_class'] == stage_group]['risk_bin'].value_counts()
                counts = counts.reindex(risk_bins_sorted, fill_value=0)
                plt.bar(x + i*width, counts.values, width, 
                    label=stage_group, alpha=0.8, color=colors[i % len(colors)])
            
            plt.xlabel(xlabel)
            plt.ylabel('Number of Patients')
            plt.title(plot_title)
            plt.xticks(x + width/2, x_labels, rotation=45)
            plt.legend(title='CKD Stage Group')
            plt.tight_layout()
            
            # Print summary statistics
            print(f"\n=== Risk Distribution Summary ({risk_bin_type} format) ===")
            summary = df.groupby(['risk_bin', 'true_class']).size().unstack(fill_value=0)
            summary = summary.reindex(risk_bins_sorted)
            summary['Total'] = summary.sum(axis=1)
            if len(summary.columns) >= 3:  # Ensure we have at least 2 stage groups + Total
                summary['% High Stage'] = (summary.iloc[:, 1] / summary['Total'] * 100).round(1)
            print(summary)
            
            # Save plot
            if self.config.get('save_plots', True):
                logger.info("Saving risk distribution plot:")
                output_dir = self.get_output_dir()
                suffix = f"_{risk_bin_type}" if risk_bin_type != 'auto' else ""
                if matched:
                    png_path = output_dir / f"risk_distribution_balanced{suffix}.png"
                else:
                    png_path = output_dir / f"risk_distribution{suffix}.png"
                plt.savefig(png_path, dpi=300, bbox_inches='tight')
                logger.info(f"Risk distribution plot saved to {png_path}")
                
        except Exception as e:
            logger.error(f"Error creating risk distribution plot: {e}")       
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        models = {}
        
        logger.info("Training models...")
        
        # Logistic Regression
        if 'logistic' in self.config.get('models', ['logistic']):
            logger.info("Training Logistic Regression...")
            models['logistic'] = model_pipeline.get_logistic_pipeline(
                df=X_train, 
                categorical_cols=self.CAT_COLUMNS, 
                numeric_cols=self.TRUE_NUM_COLUMNS,
                C=self.config.get('logistic_C', 1),
                max_iter=self.config.get('logistic_max_iter', 1000)
            )
            models['logistic'].fit(X_train, y_train)
        
        # Random Forest
        if 'random_forest' in self.config.get('models', ['random_forest']):
            logger.info("Training Random Forest...")
            models['random_forest'] = model_pipeline.get_random_forest_pipeline(
                df=X_train,
                categorical_cols=self.CAT_COLUMNS,
                numeric_cols=self.TRUE_NUM_COLUMNS
            )
            models['random_forest'].fit(X_train, y_train)
        
        # Gradient Boosting
        if 'gradient_boosting' in self.config.get('models', ['gradient_boosting']):
            logger.info("Training Gradient Boosting...")
            models['gradient_boosting'] = model_pipeline.get_histgb_pipeline(
                df=X_train,
                categorical_cols=self.CAT_COLUMNS,
                numeric_cols=self.TRUE_NUM_COLUMNS,
                balance_classes=self.config.get('balance_classes', True)
            )
            models['gradient_boosting'].fit(X_train, y_train)
        
        # IAI OCT (if available)
        if 'oct' in self.config.get('models', []):
            logger.info("Training OCT...")
            try:
                iai_model, preprocessor, feature_names = train_oct_with_feature_names(
                    X_train, y_train, 
                    categorical_cols=self.CAT_COLUMNS,
                    numeric_cols=self.TRUE_NUM_COLUMNS
                )
                models['oct'] = {'model': iai_model, 'preprocessor': preprocessor, 
                               'feature_names': feature_names}
            except Exception as e:
                logger.warning(f"OCT training failed: {e}")
        
        # IAI OPT (if available)
        if 'opt' in self.config.get('models', []):
            logger.info("Training OPT...")
            try:
                stratifier_cutoff = self.config.get('stratifier_cutoff', 3)
                X_no_stage = X_train.drop(columns=self.STAGE_COLUMNS, errors='ignore')
                treatments = (X_train["stage_2017"] > 3).astype(int)
                
                # Use a default cutoff column for OPT outcomes or make it configurable
                default_outcome_col = self.config.get('opt_outcome_column', 'highcost_gt_100000')
                outcomes = self.df_og.loc[X_train.index, default_outcome_col].astype(int)
                
                opt_learner, preprocessor, feature_names = train_opt_with_feature_names(
                    X_no_stage, treatments, outcomes,
                    categorical_cols=self.CAT_COLUMNS,
                    numeric_cols=self.TRUE_NUM_COLUMNS
                )
                models['opt'] = {'model': opt_learner, 'preprocessor': preprocessor,
                               'feature_names': feature_names}
            except Exception as e:
                logger.warning(f"OPT training failed: {e}")
        
        return models

    def apply_matching(self, train_df):
        if not self.config.get('apply_matching', True):
            logger.info("Matching disabled in config")
            return train_df, []
        
        try:
            logger.info("Applying risk-binned case-control matching...")
            
            # Ensure binary group column exists
            train_df = train_df.copy()
            if "true_class" not in train_df.columns:
                train_df["true_class"] = (train_df["stage_2017"] > self.stratifier_stage_cutoff).astype(int)
            
            # Initialize the resampler
            from balancing_functions.optimal_match_control import EnhancedRiskBinnedCaseControlResampler, get_high_cost_detection_config
            
            sampler = EnhancedRiskBinnedCaseControlResampler(
                matching_method=self.config.get('matching_method', 'ortools'),
                random_state=self.config.get('random_seed', 42),
                proba_col="risk_score",
                binary_group="true_class",
                uid_col="ENROLID"
            )
            
            # Get matching strategies
            strategies = get_high_cost_detection_config(
                primary_matching_method=self.config.get('matching_method', 'ortools')
            )
            
            # Calculate target bin size based on minimum size of true_class=1 bins
            bin_counts_by_class = train_df.groupby(['risk_bin', 'true_class']).size().unstack(fill_value=0)
            if 1 in bin_counts_by_class.columns:
                target_size = bin_counts_by_class[1].min()
                logger.info(f"Setting target size to minimum stage>{self.stratifier_stage_cutoff} bin size: {target_size}")
            else:
                logger.warning("No stage>{self.stratifier_stage_cutoff} cases found, using overall minimum bin size")
                target_size = train_df["risk_bin"].value_counts().min()
            logger.info(f"Bin sizes before matching: {train_df.risk_bin.value_counts().to_dict()}")
            
            # Apply matching
            matched_train_df, removed_ids = sampler.apply_sampler_by_bin(
                train_df,
                exclude_cols_matching= self.TRUE_NUM_COLUMNS+self.STAGE_COLUMNS,
                bin_specific_strategies=strategies,
                target_bin_size=target_size,
                verbose=self.config.get('verbose', True)
            )
            
            logger.info(f"Training set size: {len(train_df)} -> {len(matched_train_df)}")
            
            return matched_train_df, removed_ids
            
        except ImportError:
            logger.error("Matching modules not available. Install balancing_functions package.")
            return train_df, []
        except Exception as e:
            logger.error(f"Matching failed: {e}")
            return train_df, []     
    
    def evaluate_models(self, models, X_test, y_test, test_df, cutoff_colname, matching_status="before"):
        """Evaluate all trained models"""
        results = []
        
        # Extract cutoff value from column name for backwards compatibility
        cutoff_value = int(cutoff_colname.split('_')[-1])
        
        
        for model_name, model in models.items():
            try:
                if model_name in ['oct']:
                    logger.info(f"Evaluating models for {cutoff_colname}, matching={matching_status}")
                    X_test_transformed = model['preprocessor'].transform(X_test)
                    X_test_df = pd.DataFrame(X_test_transformed, columns=model['feature_names'])
                    result = evaluate_metrics_by_risk_bin(
                        model['model'], X_test_df, y_test, test_df['risk_bin'],
                        model_name=model_name,
                        dataset_name=f'{matching_status.title()} Matching'
                    )
                else:
                    logger.info(f"Evaluating models for {cutoff_colname}, matching={matching_status}")
                    # Handle sklearn models
                    result = evaluate_metrics_by_risk_bin(
                        model, X_test, y_test, test_df['risk_bin'],
                        model_name=model_name.title().replace('_', ' '),
                        dataset_name=f'{matching_status.title()} Matching'
                    )
                
                result['cutoff_value'] = cutoff_value
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
                
    def run_single_experiment(self, cutoff_colname):
        """Run a single experiment with a given cutoff column"""
        logger.info(f"Starting experiment with {cutoff_colname}")
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[cutoff_colname]
        # Train and evaluate on original data
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[cutoff_colname]
        
        models_original = self.train_models(X_train, y_train)
        results_before = self.evaluate_models(
            models_original, X_test, y_test, self.test_df, cutoff_colname, "before"
        )
        # Train and evaluate on matched data if available
        if self.matched_train_df is not None:
            X_train_matched = self.matched_train_df[self.feature_cols]
            y_train_matched = self.matched_train_df[cutoff_colname]
            models_matched = self.train_models(X_train_matched, y_train_matched)
            results_after = self.evaluate_models(
                models_matched, X_test, y_test, self.test_df, cutoff_colname, "after"
            )
            all_results = pd.concat([results_before, results_after], ignore_index=True)
        else:
            all_results = results_before
        
        return all_results,  {'original': models_original, 'matched': models_matched if self.matched_train_df is not None else None}
        
    def run_pipeline(self):
        """Run the complete pipeline for all cutoff columns"""
        logger.info("Starting ML Pipeline...")
        
        # Load data
        self.load_and_prepare_data()
        self.feature_cols = self.prepare_features()
        # Create risk bins once (independent of cutoff)
        self.df_with_risk_bins = self.create_risk_bins()
         # Use a dummy train split just for matching setup
        _,_,self.train_df, self.test_df  = model_pipeline.train_test_split_enrol(
            df=self.df_with_risk_bins, target_col='highcost_gt_50000',  # Use any cutoff for splitting.
            random_state = self.config.get('random_seed', 42)
        )
        self.plot_risk_distribution(self.train_df, matched=False)
        output_dir = self.get_output_dir()
        # Apply matching once if enabled
        if self.config.get('apply_matching', True):
            logger.info("Applying matching to full dataset...")
            self.matched_train_df, self.removed_ids = self.apply_matching(self.train_df)
            logger.info(f"Matching completed: {len(self.train_df)} -> {len(self.matched_train_df)}")
            logger.info(f"Saving matched train set to: {output_dir}")
            # Save results
            matched_file = output_dir / f"matched_train_df.csv"
            self.matched_train_df.to_csv(matched_file,index=False)
            self.plot_risk_distribution(self.matched_train_df, matched=True)

        else:
            logger.info(f"Loading matched train set from: {output_dir}")
            matched_file = output_dir / f"matched_train_df.csv"
            self.matched_train_df = pd.read_csv(matched_file)
            
        # Get cutoff columns to test
        if 'cutoff_colnames' in self.config:
            cutoff_colnames = self.config['cutoff_colnames']
        else:
            cutoff_colnames = self.cutoff_colnames
        
        # Run experiments for each cutoff column
        for cutoff_colname in cutoff_colnames:
            try:
                results, models = self.run_single_experiment(cutoff_colname)
                self.results.append(results)
                self.models[cutoff_colname] = models
                
                logger.info(f"Completed experiment for {cutoff_colname}")
                
            except Exception as e:
                logger.error(f"Error in experiment for {cutoff_colname}: {e}")
                continue
        
        # Combine all results
        if self.results:
            final_results = pd.concat(self.results, ignore_index=True)
            
            # Save results
            self.save_results(final_results)
            
            return final_results, self.models
        else:
            logger.error("No experiments completed successfully")
            return pd.DataFrame(), {}
            
    def save_results(self, results_df):
        """Save results and models"""
        logger.info(f"Config output_dir: {self.config.get('output_dir')}")
                
        output_dir = self.get_output_dir()
        logger.info(f"Resolved output_dir: {output_dir}")
                
        # Save results
        results_file = output_dir / f"pipeline_results.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")
        
        # Save matching summary if matching was applied
        if hasattr(self, 'matched_train_df') and self.matched_train_df is not None:
            matching_summary = {
                'matching_applied': True,
                'original_training_size': len(self.df_with_risk_bins),
                'matched_training_size': len(self.matched_train_df),
                'reduction_percentage': (1 - len(self.matched_train_df) / len(self.df_with_risk_bins)) * 100,
                'removed_ids_count': sum(len(ids) for ids in self.removed_ids.values()) if hasattr(self, 'removed_ids') else 0
            }
            
            matching_file = output_dir / f"matching_summary.json"
            with open(matching_file, 'w') as f:
                json.dump(matching_summary, f, indent=2)
            logger.info(f"Matching summary saved to {matching_file}")
        
        # Save results summary
        if not results_df.empty:
            summary_stats = results_df.groupby(['dataset_name', 'cutoff_value', 'model_name']).agg({
                'auc': 'mean',
                'precision': 'mean', 
                'recall': 'mean',
                'f1': 'mean'
            }).round(4)
            
            summary_file = output_dir / f"results_summary.csv"
            summary_stats.to_csv(summary_file)
            logger.info(f"Results summary saved to {summary_file}")
                
        logger.info(f"Pipeline completed. Results available in {output_dir}")
    def cleanup(self):
        """Clean up Spark session"""
        try:
            spark.stop()
            logger.info("Spark session stopped")
        except Exception as e:
            logger.warning(f"Error stopping Spark session: {e}")

def load_config(config_path=None):
    """Load configuration from file or return default"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            'data_path': '0813_cost_features_2017.parquet',
            'cutoff_colnames': ['highcost_gt_50000', 'highcost_gt_100000', 'highcost_gt_200000'],
            'stratifier_stage_cutoff': 3,
            'test_size': 0.3,
            'models': ['logistic', 'random_forest', 'gradient_boosting'],
            'logistic_C': 1,
            'logistic_max_iter': 1000,
            'balance_classes': True,
            'output_dir': 'output'
        }

def main():
    parser = argparse.ArgumentParser(description='ML Pipeline for High-Cost Prediction')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--cutoff-columns', nargs='+', type=str, 
                       help='High-cost cutoff column names to test (overrides config)')
    parser.add_argument('--models', nargs='+', type=str,
                       choices=['logistic', 'random_forest', 'gradient_boosting', 'oct', 'opt'],
                       help='Models to train (overrides config)')
    parser.add_argument('--data-path', type=str, help='Path to data file (overrides config)')
    parser.add_argument('--output-dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--stratifier-stage-cutoff', type=int, 
                       help='Stage cutoff for stratifier training (overrides config)')
    parser.add_argument('--stratifier-model-type', type=str,
                       help='Target column for stratifier (overrides config)')
    parser.add_argument('--n_strata', type=int,
                       help='Number of predicted 2018 cost strata (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.cutoff_columns:
        config['cutoff_colnames'] = args.cutoff_columns
    if args.models:
        config['models'] = args.models
    if args.data_path:
        config['data_path'] = args.data_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.stratifier_stage_cutoff:
        config['stratifier_stage_cutoff'] = args.stratifier_stage_cutoff
    if args.stratifier_model_type:
        config['stratifier_model_type'] = args.stratifier_model_type
    if args.n_strata:
        config['n_strata'] = args.n_strata
    
    # Initialize and run pipeline
    pipeline = MLPipeline(config)
    
    try:
        results, models = pipeline.run_pipeline()
        logger.info("Pipeline execution completed!")
    finally:
        # Always cleanup Spark
        pipeline.cleanup()

if __name__ == "__main__":
    main()