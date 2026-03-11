"""
Precompute pairwise distances between majority and minority classes.
Optimized for memory efficiency and fast access.

Distance metrics: 'euclidean' (L2, default), 'manhattan' (L1), 'chebyshev' (L_infinity).
Pass metric= to compute_distances_batched() and precompute_leaf_dnn_memmap().

Usage:
    python precompute_distances.py
"""
import pandas as pd
import numpy as np
import h5py
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import time
import os
from tqdm import tqdm



def get_preprocessor(X, cat_cols, num_cols, binary_cols=None, verbose=False):
    """
    Create preprocessor matching OCT model preprocessing.
    
    This function now uses model_IAI.get_preprocessor_with_impute for consistency
    with the model training pipeline.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix (used as X_train for imputation check)
    cat_cols : list
        Categorical column names
    num_cols : list
        Numeric column names (will be scaled)
    binary_cols : list, optional
        Binary flag column names (0/1). These will be passed through
        without scaling. If None, binary columns are dropped.
    verbose : bool, default=False
        Whether to print preprocessing details
    
    Returns:
    --------
    preprocessor : ColumnTransformer
    
    Note: Uses get_preprocessor_with_impute from model_IAI for consistency
          with the model training pipeline. Binary flag columns are handled
          explicitly if provided.
    """
    # Try to import from model_IAI, fall back to local implementation if not available
    try:
        # Try multiple import paths for flexibility
        try:
            from public.model_IAI import get_preprocessor_with_impute
        except ImportError:
            try:
                import model_IAI
                get_preprocessor_with_impute = model_IAI.get_preprocessor_with_impute
            except ImportError:
                # Try relative import
                from .model_IAI import get_preprocessor_with_impute
        
        return get_preprocessor_with_impute(
            X_train=X,
            categorical_cols=cat_cols,
            numeric_cols=num_cols,
            binary_cols=binary_cols,
            verbose=verbose
        )
    except (ImportError, AttributeError) as e:
        # Fallback to original implementation if model_IAI is not available
        print(f"Error importing get_preprocessor_with_impute: {e}")
        # This maintains backward compatibility
        # IMPORTANT: Filter columns to only those present in X to avoid KeyErrors
        cat_cols_filtered = [col for col in cat_cols if col in X.columns] if cat_cols else []
        num_cols_filtered = [col for col in num_cols if col in X.columns] if num_cols else []
        binary_cols_filtered = [col for col in binary_cols if col in X.columns] if binary_cols else []
        
        transformers = []
        if cat_cols_filtered:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols_filtered))
        if num_cols_filtered:
            transformers.append(('num', StandardScaler(), num_cols_filtered))
        if binary_cols_filtered:
            # Binary columns pass through unchanged
            transformers.append(('binary', FunctionTransformer(), binary_cols_filtered))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Keep remaining columns (e.g., binary flags) unchanged
        )
        return preprocessor


def compute_distances_batched(X_majority, X_minority, batch_size=1000, dtype=np.float32, metric='euclidean'):
    """
    Compute pairwise distances in batches to manage memory
    
    Parameters:
    -----------
    X_majority : array (n_majority, n_features)
    X_minority : array (n_minority, n_features)
    batch_size : int, number of majority samples per batch
    dtype : data type for distances (float32 recommended)
    metric : str, distance metric. One of 'euclidean' (L2), 'manhattan' (L1),
             'chebyshev' (L_infinity). Default 'euclidean'.
    
    Returns:
    --------
    distances : array (n_majority, n_minority) with dtype
    """
    n_majority = X_majority.shape[0]
    n_minority = X_minority.shape[0]
    
    # Pre-allocate output array
    distances = np.zeros((n_majority, n_minority), dtype=dtype)
    
    print(f"Computing {n_majority:,} x {n_minority:,} = {n_majority * n_minority:,} distances (metric={metric})")
    print(f"Output size: {distances.nbytes / 1e6:.1f} MB ({dtype})")
    
    # Process in batches
    n_batches = (n_majority + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Computing distances"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_majority)
        
        batch_distances = pairwise_distances(
            X_majority[start_idx:end_idx], 
            X_minority,
            metric=metric
        )
        
        distances[start_idx:end_idx] = batch_distances.astype(dtype)
    
    return distances


def save_distances_hdf5(distances, majority_ids, minority_ids, filepath, compression='gzip', compression_opts=9):
    """
    Save distances in HDF5 format with metadata (chunked + compressed).
    HDF5 supports reading directly from compressed format (no separate unzip needed).
    Use compression_opts=9 for max compression; 4 for faster write/read.

    File structure:
        /distances: (n_majority, n_minority) float32 array
        /majority_enrolids: (n_majority,) int64 array
        /minority_enrolids: (n_minority,) int64 array
        /metadata: attributes with shape, dtype, etc.
    """
    print(f"\nSaving to HDF5: {filepath} (compression={compression}, level={compression_opts})")
    
    with h5py.File(filepath, 'w') as f:
        chunk_size = (min(1000, distances.shape[0]), distances.shape[1])
        f.create_dataset(
            'distances',
            data=distances,
            chunks=chunk_size,
            compression=compression,
            compression_opts=compression_opts
        )
        
        # Save ENROLID mappings
        f.create_dataset('majority_enrolids', data=majority_ids)
        f.create_dataset('minority_enrolids', data=minority_ids)
        
        # Metadata
        f.attrs['n_majority'] = distances.shape[0]
        f.attrs['n_minority'] = distances.shape[1]
        f.attrs['dtype'] = str(distances.dtype)
        f.attrs['compression'] = compression
    
    file_size = os.path.getsize(filepath) / 1e6
    print(f"  ✓ Saved {file_size:.1f} MB")


def save_distances_numpy_memmap(distances, majority_ids, minority_ids, base_filepath):
    """
    Save distances as numpy memmap (fastest access, no compression)
    
    Creates 3 files:
        - {base}_distances.npy: (n_majority, n_minority) distance matrix
        - {base}_majority_ids.npy: (n_majority,) ENROLID array
        - {base}_minority_ids.npy: (n_minority,) ENROLID array
    """
    print(f"\nSaving to Numpy memmap: {base_filepath}")
    
    # Save distances
    distances_path = f"{base_filepath}_distances.npy"
    np.save(distances_path, distances)
    
    # Save ID mappings
    np.save(f"{base_filepath}_majority_ids.npy", majority_ids)
    np.save(f"{base_filepath}_minority_ids.npy", minority_ids)
    
    total_size = (
        os.path.getsize(distances_path) +
        os.path.getsize(f"{base_filepath}_majority_ids.npy") +
        os.path.getsize(f"{base_filepath}_minority_ids.npy")
    ) / 1e6
    
    print(f"  ✓ Saved {total_size:.1f} MB (3 files)")




def precompute_leaf_dnn_memmap(X_majority_leaf: np.ndarray,
                               majority_enrolids_leaf: np.ndarray,
                               out_dir: str,
                               leaf_id: str,
                               batch_size: int = 750,
                               dtype=np.float32,
                               metric: str = 'euclidean'):
    """
    metric : str
        Distance metric: 'euclidean' (L2), 'manhattan' (L1), or 'chebyshev' (L_infinity).
    """
    os.makedirs(out_dir, exist_ok=True)
    n = X_majority_leaf.shape[0]

    dnn_matrix_path = os.path.join(out_dir, f"leaf_{leaf_id}_dnn_matrix.npy")
    dnn_enrolids_path = os.path.join(out_dir, f"leaf_{leaf_id}_dnn_enrolids.npy")

    print(f"[leaf {leaf_id}] computing d_nn for n={n:,} -> ~{n*n*4/1e9:.2f} GB float32 (metric={metric})")

    # Create a writable .npy memmap (so we can fill row blocks)
    dnn_mm = np.lib.format.open_memmap(dnn_matrix_path, mode="w+", dtype=dtype, shape=(n, n))

    n_batches = (n + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc=f"leaf {leaf_id} d_nn"):
        s = b * batch_size
        e = min((b + 1) * batch_size, n)
        dblock = pairwise_distances(X_majority_leaf[s:e], X_majority_leaf, metric=metric)
        dnn_mm[s:e, :] = dblock.astype(dtype, copy=False)

    # Flush to disk
    del dnn_mm

    np.save(dnn_enrolids_path, majority_enrolids_leaf.astype(np.int64, copy=False))
    print(f"[leaf {leaf_id}] saved:\n  {dnn_matrix_path}\n  {dnn_enrolids_path}")
    return dnn_matrix_path, dnn_enrolids_path


def precompute_leaf_dnn_hdf5(
    X_majority_leaf: np.ndarray,
    majority_enrolids_leaf: np.ndarray,
    out_dir: str,
    leaf_id: str,
    batch_size: int = 750,
    dtype=np.float32,
    metric: str = 'euclidean',
    compression: str = 'gzip',
    compression_opts: int = 9,
):
    """
    Same as precompute_leaf_dnn_memmap but saves to HDF5 with chunking + compression.
    Much smaller on disk (often 3-5x) than raw .npy. Read via h5py; slicing (e.g. d[i,:])
    decompresses only needed chunks on the fly.

    Returns (h5_path, enrolids_path) where enrolids_path is .npy (small, uncompressed).
    """
    os.makedirs(out_dir, exist_ok=True)
    n = X_majority_leaf.shape[0]
    h5_path = os.path.join(out_dir, f"leaf_{leaf_id}_dnn_matrix.h5")
    enrolids_path = os.path.join(out_dir, f"leaf_{leaf_id}_dnn_enrolids.npy")

    print(f"[leaf {leaf_id}] computing d_nn for n={n:,} -> HDF5 compressed (metric={metric})")

    with h5py.File(h5_path, 'w') as f:
        chunk_rows = min(1000, n)
        dset = f.create_dataset(
            'distances',
            shape=(n, n),
            dtype=dtype,
            chunks=(chunk_rows, n),
            compression=compression,
            compression_opts=compression_opts,
        )
        n_batches = (n + batch_size - 1) // batch_size
        for b in tqdm(range(n_batches), desc=f"leaf {leaf_id} d_nn"):
            s = b * batch_size
            e = min((b + 1) * batch_size, n)
            dblock = pairwise_distances(X_majority_leaf[s:e], X_majority_leaf, metric=metric)
            dset[s:e, :] = dblock.astype(dtype, copy=False)

    np.save(enrolids_path, majority_enrolids_leaf.astype(np.int64, copy=False))
    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"[leaf {leaf_id}] saved: {h5_path} ({size_mb:.1f} MB), {enrolids_path}")
    return h5_path, enrolids_path

