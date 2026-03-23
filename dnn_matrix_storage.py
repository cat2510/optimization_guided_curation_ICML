"""
Paths and helpers for Gower D-N-N matrices on disk (uncompressed .npy or .npz + mmap).

D-N-N is stored as leaf_global_dnn_matrix.npy or leaf_global_dnn_matrix.npz for fast
random access; do not gzip (ratio is poor on float distances and compress/decompress
is very slow).
"""
from __future__ import annotations

import os

LEAF_GLOBAL_DNN_MATRIX_NPY = "leaf_global_dnn_matrix.npy"
LEAF_GLOBAL_DNN_MATRIX_NPZ = "leaf_global_dnn_matrix.npz"
LEAF_GLOBAL_DNN_ENROLIDS_NPY = "leaf_global_dnn_enrolids.npy"


def dnn_matrix_npy_path(dnn_dir: str) -> str:
    return os.path.join(os.path.abspath(dnn_dir), LEAF_GLOBAL_DNN_MATRIX_NPY)


def dnn_matrix_path(dnn_dir: str, fmt: str = "npy") -> str:
    """Return path to .npy or .npz based on fmt."""
    if fmt == "npz":
        return os.path.join(os.path.abspath(dnn_dir), LEAF_GLOBAL_DNN_MATRIX_NPZ)
    return dnn_matrix_npy_path(dnn_dir)


def dnn_enrolids_npy_path(dnn_dir: str) -> str:
    return os.path.join(os.path.abspath(dnn_dir), LEAF_GLOBAL_DNN_ENROLIDS_NPY)


def dnn_matrix_storage_exists(dnn_dir: str) -> bool:
    """True if leaf_global_dnn_matrix.npy or leaf_global_dnn_matrix.npz exists under dnn_dir."""
    base = os.path.abspath(dnn_dir)
    return os.path.isfile(os.path.join(base, LEAF_GLOBAL_DNN_MATRIX_NPY)) or os.path.isfile(
        os.path.join(base, LEAF_GLOBAL_DNN_MATRIX_NPZ)
    )


def ensure_dnn_matrix_npy(dnn_dir: str) -> str:
    """
    Return absolute path to leaf_global_dnn_matrix.npy or .npz (for np.load(..., mmap_mode='r')).
    Checks .npy first, then .npz. Raises FileNotFoundError if neither exists.
    """
    base = os.path.abspath(dnn_dir)
    p_npy = os.path.join(base, LEAF_GLOBAL_DNN_MATRIX_NPY)
    p_npz = os.path.join(base, LEAF_GLOBAL_DNN_MATRIX_NPZ)
    if os.path.isfile(p_npy):
        return p_npy
    if os.path.isfile(p_npz):
        return p_npz
    raise FileNotFoundError(f"Missing D-N-N matrix: {p_npy} or {p_npz}")
