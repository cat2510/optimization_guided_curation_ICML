"""
debug_id_alignment.py
=====================
Minimal diagnostic module for verifying ENROLID alignment between
precomputed distance artifacts (d_nn, d_pn) and the train-split IDs
used at experiment time.

Usage (standalone smoke test):
    python -m public.debug_id_alignment \
        --distances_dir ./precomputed_distances_msk_medical_only \
        --seed 123

All functions are pure diagnostics — they never mutate data.
"""
from __future__ import annotations

import numpy as np
import h5py
from typing import Optional


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sorted_sample(ids: np.ndarray, n: int = 5) -> list:
    s = np.sort(ids)
    head = s[:n].tolist()
    tail = s[-n:].tolist()
    return head, tail


def fingerprint(name: str, ids: np.ndarray) -> str:
    """Return a one-line fingerprint: min/max/len + first/last 5 sorted IDs."""
    s = np.sort(ids)
    head = s[:5].tolist()
    tail = s[-5:].tolist()
    return (
        f"  {name}: len={len(ids)}, min={int(s[0])}, max={int(s[-1])}, "
        f"head5={head}, tail5={tail}"
    )


# ------------------------------------------------------------------
# summarize_overlap
# ------------------------------------------------------------------

def summarize_overlap(nameA: str, idsA: np.ndarray,
                      nameB: str, idsB: np.ndarray) -> dict:
    """Print counts / % overlap between two ID arrays and return stats dict."""
    setA = set(int(x) for x in idsA)
    setB = set(int(x) for x in idsB)
    inter = setA & setB
    only_A = setA - setB
    only_B = setB - setA

    pct_A_in_B = 100.0 * len(inter) / max(len(setA), 1)
    pct_B_in_A = 100.0 * len(inter) / max(len(setB), 1)

    print(f"  Overlap: {nameA} ({len(setA):,}) vs {nameB} ({len(setB):,})")
    print(f"    intersection = {len(inter):,}  "
          f"({pct_A_in_B:.2f}% of {nameA}, {pct_B_in_A:.2f}% of {nameB})")
    if only_A:
        sample = sorted(only_A)[:10]
        print(f"    only in {nameA}: {len(only_A):,}  sample={sample}")
    if only_B:
        sample = sorted(only_B)[:10]
        print(f"    only in {nameB}: {len(only_B):,}  sample={sample}")

    return {
        "intersection": len(inter),
        "only_A": len(only_A),
        "only_B": len(only_B),
        "pct_A_in_B": pct_A_in_B,
        "pct_B_in_A": pct_B_in_A,
    }


# ------------------------------------------------------------------
# assert_subset
# ------------------------------------------------------------------

def assert_subset(nameA: str, idsA: np.ndarray,
                  nameB: str, idsB: np.ndarray,
                  allow_missing: bool = False) -> None:
    """Assert set(idsA) ⊆ set(idsB).  Raise with first ~10 offenders."""
    setA = set(int(x) for x in idsA)
    setB = set(int(x) for x in idsB)
    missing = setA - setB
    if missing:
        sample = sorted(missing)[:10]
        msg = (
            f"assert_subset FAILED: {len(missing):,} IDs in {nameA} "
            f"not found in {nameB}.  First 10: {sample}"
        )
        if allow_missing:
            print(f"  WARNING: {msg}")
        else:
            raise AssertionError(msg)
    else:
        print(f"  OK: {nameA} ({len(setA):,}) ⊆ {nameB} ({len(setB):,})")


# ------------------------------------------------------------------
# check_dnn_files
# ------------------------------------------------------------------

def check_dnn_files(dnn_ids: np.ndarray,
                    control_enrolids: np.ndarray,
                    d_nn_path: str,
                    n_spot: int = 5) -> None:
    """
    Load d_nn (mmap), verify:
      * square and len matches dnn_ids
      * d_nn[i,i] ≈ 0  for random i
      * d_nn[i,j] ≈ d_nn[j,i]  (symmetry)
      * overlap of dnn_ids vs control_enrolids
    """
    print("\n" + "=" * 60)
    print("check_dnn_files")
    print("=" * 60)

    d_nn = np.load(d_nn_path, mmap_mode="r")
    print(f"  d_nn shape: {d_nn.shape}, dtype: {d_nn.dtype}")
    print(f"  dnn_ids len: {len(dnn_ids)}")

    # Shape checks
    assert d_nn.shape[0] == d_nn.shape[1], (
        f"d_nn not square: {d_nn.shape}"
    )
    assert d_nn.shape[0] == len(dnn_ids), (
        f"d_nn rows ({d_nn.shape[0]}) != len(dnn_ids) ({len(dnn_ids)})"
    )

    # Spot checks
    rng = np.random.RandomState(42)
    n = d_nn.shape[0]
    idxs = rng.choice(n, size=min(n_spot, n), replace=False)
    print(f"\n  Spot-check diagonal (should be ≈ 0):")
    for i in idxs:
        val = float(d_nn[i, i])
        status = "OK" if abs(val) < 1e-6 else f"WARN diag≠0 ({val})"
        print(f"    d_nn[{i},{i}] = {val:.8f}  [{status}]")

    print(f"\n  Spot-check symmetry:")
    pairs = list(zip(idxs[:-1], idxs[1:]))
    for i, j in pairs:
        vij = float(d_nn[i, j])
        vji = float(d_nn[j, i])
        diff = abs(vij - vji)
        status = "OK" if diff < 1e-5 else f"WARN |diff|={diff:.8f}"
        print(f"    d_nn[{i},{j}]={vij:.6f}  d_nn[{j},{i}]={vji:.6f}  [{status}]")

    # Overlap
    print()
    summarize_overlap("dnn_ids", dnn_ids, "control_enrolids", control_enrolids)
    print(fingerprint("dnn_ids", dnn_ids))
    print(fingerprint("control_enrolids", control_enrolids))


# ------------------------------------------------------------------
# check_pn_files
# ------------------------------------------------------------------

def check_pn_files(pn_h5_path: str,
                   control_enrolids: np.ndarray,
                   case_enrolids: np.ndarray,
                   dnn_ids: Optional[np.ndarray] = None,
                   n_spot: int = 5) -> None:
    """
    Load pn HDF5, verify:
      * pn_maj_ids overlap with control_enrolids
      * pn_min_ids overlap with case_enrolids
      * spot-check: 5 random eid from dnn_ids ∩ pn_maj_ids
        → pn_maj_id2idx[eid] exists and indexing returns expected row shape
    """
    print("\n" + "=" * 60)
    print("check_pn_files")
    print("=" * 60)

    f = h5py.File(pn_h5_path, "r")
    try:
        d_pn = f["distances"]
        pn_maj_ids = f["majority_enrolids"][:]
        pn_min_ids = f["minority_enrolids"][:]

        print(f"  d_pn shape: {d_pn.shape}, dtype: {d_pn.dtype}")
        print(f"  pn_maj_ids len: {len(pn_maj_ids)}")
        print(f"  pn_min_ids len: {len(pn_min_ids)}")

        assert d_pn.shape[0] == len(pn_maj_ids), (
            f"d_pn rows ({d_pn.shape[0]}) != pn_maj_ids ({len(pn_maj_ids)})"
        )
        assert d_pn.shape[1] == len(pn_min_ids), (
            f"d_pn cols ({d_pn.shape[1]}) != pn_min_ids ({len(pn_min_ids)})"
        )

        # Overlap stats
        print()
        summarize_overlap("pn_maj_ids", pn_maj_ids,
                          "control_enrolids", control_enrolids)
        print()
        summarize_overlap("pn_min_ids", pn_min_ids,
                          "case_enrolids", case_enrolids)

        print()
        print(fingerprint("pn_maj_ids", pn_maj_ids))
        print(fingerprint("pn_min_ids", pn_min_ids))
        print(fingerprint("control_enrolids", control_enrolids))
        print(fingerprint("case_enrolids", case_enrolids))

        # Spot-check indexing for shared IDs
        if dnn_ids is not None:
            pn_maj_set = set(int(x) for x in pn_maj_ids)
            dnn_set = set(int(x) for x in dnn_ids)
            common = sorted(dnn_set & pn_maj_set)
            if common:
                pn_maj_id2idx = {int(x): i for i, x in enumerate(pn_maj_ids)}
                rng = np.random.RandomState(42)
                sample = rng.choice(common,
                                    size=min(n_spot, len(common)),
                                    replace=False)
                print(f"\n  Spot-check indexing (dnn_ids ∩ pn_maj_ids, n={len(common):,}):")
                for eid in sample:
                    eid = int(eid)
                    idx = pn_maj_id2idx[eid]
                    row = np.array(d_pn[idx, :5], dtype=np.float32)
                    print(f"    eid={eid} → pn_row_idx={idx}, "
                          f"row[:5]={row.tolist()}, shape_ok={d_pn[idx].shape}")
            else:
                print("  WARNING: dnn_ids ∩ pn_maj_ids is EMPTY")
    finally:
        f.close()


# ------------------------------------------------------------------
# Full alignment report
# ------------------------------------------------------------------

def run_full_alignment_check(
    dnn_ids: np.ndarray,
    control_enrolids: np.ndarray,
    case_enrolids: np.ndarray,
    d_nn_path: str,
    pn_h5_path: str,
    train_test_seed: int,
    distances_dir: str,
    strict: bool = True,
) -> None:
    """
    Run all alignment checks and print a summary fingerprint.

    Parameters
    ----------
    strict : bool
        If True, raise on subset violations instead of warning.
    """
    print("\n" + "#" * 70)
    print("  DEBUG ALIGNMENT REPORT")
    print("#" * 70)
    print(f"  TRAIN_TEST_SEED  : {train_test_seed}")
    print(f"  distances_dir    : {distances_dir}")
    print()
    print(fingerprint("dnn_ids", dnn_ids))
    print(fingerprint("control_enrolids", control_enrolids))
    print(fingerprint("case_enrolids", case_enrolids))

    # 1. d_nn checks
    check_dnn_files(dnn_ids, control_enrolids, d_nn_path)

    # 2. d_pn checks
    check_pn_files(pn_h5_path, control_enrolids, case_enrolids,
                   dnn_ids=dnn_ids)

    # 3. Hard subset assertions
    print("\n" + "=" * 60)
    print("Hard subset assertions")
    print("=" * 60)
    assert_subset("dnn_ids", dnn_ids, "control_enrolids", control_enrolids,
                  allow_missing=not strict)

    f = h5py.File(pn_h5_path, "r")
    try:
        pn_maj_ids = f["majority_enrolids"][:]
        pn_min_ids = f["minority_enrolids"][:]
    finally:
        f.close()

    assert_subset("case_enrolids", case_enrolids, "pn_min_ids", pn_min_ids,
                  allow_missing=not strict)
    assert_subset("control_enrolids", control_enrolids, "pn_maj_ids", pn_maj_ids,
                  allow_missing=not strict)

    print("\n" + "#" * 70)
    print("  ALIGNMENT CHECK COMPLETE")
    print("#" * 70 + "\n")
