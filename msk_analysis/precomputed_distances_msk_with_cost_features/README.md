    
*** Only cost columns (no 2018) ***
DISTANCES_DIR = "./precomputed_distances_msk_cost_only",
        exclude_cols = [col for col in train_pd.columns if col not in COST_COLUMNS] 
        drop_cols = ['ENROLID']+ exclude_cols


*** No cost features no 2018 features ***
DISTANCES_DIR = "./precomputed_distances_msk_medical_only"
        exclude_cols = COST_COLUMNS + [col for col in train_pd.columns if "2018" in col] 
        drop_cols = ['ENROLID']+ exclude_cols


*** All features but no 2018 features ***
DISTANCES_DIR = "./precomputed_distances_msk_with_cost_features"
        exclude_cols = [col for col in train_pd.columns if "2018" in col] 
        drop_cols = ['ENROLID']+ exclude_cols

*** Medical + 2018 features ***
DISTANCES_DIR = "./precomputed_distances_msk_with_target_no_cost",
        exclude_cols = COST_COLUMNS
        drop_cols = ['ENROLID']+ exclude_cols


*** Medical + direct MSK cost features ***
DISTANCES_DIR = "./precomputed_distances_msk_less_cost"