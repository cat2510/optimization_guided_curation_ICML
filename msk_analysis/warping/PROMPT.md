Create a new exploratory Jupyter notebook in my_projects/msk_analysis/ to test sampling-time feature warping for deterministic Stage A k-center selection on a small subset of the MSK data.

Goal:
I want a compact notebook that uses a small stratified sample of the imbalanced dataset and a small hand-picked set of mixed-type features to compare several candidate feature warps for Stage A sampling. The warp is ONLY for sampling geometry, not for downstream prediction features. This notebook is exploratory and diagnostic, not the full experiment pipeline.

Create a notebook with a clear narrative, comments, and plots.
Suggested notebook path:
- /Users/cat2510/my_projects/msk_analysis/warping/stageA_warp_sandbox.ipynb

Dataset:
- Use the local parquet:
  /Users/cat2510/my_projects/msk_analysis/msk_2017_18_full.parquet
If the exact path differs in the repo, search for the parquet and use the correct local path.

Main tasks for the notebook:

1. Load data and define the target
- Inspect columns and identify the target consistent with the main project pipeline.
- If needed, refer to the existing experiment scripts in:
  /Users/cat2510/my_projects/msk_analysis/scripts/ like @precompute_msk_gower_distances.py (73-74) to match the target definition used there.
- Print basic dataset shape and class balance.

2. Create a small stratified subsample
- Randomly subsample the full dataset while preserving the target distribution.
- Make the sample small enough for quick experimentation in notebook form.
- Default target size: 5,000 rows
- Print the class counts and minority prevalence in:
  - full dataset
  - subsample

3. Define a small mixed-type feature set
- For now, use a hand-picked list of 4 predictive mixed-type features that are plausible from prior random OCTs.
- IMPORTANT: first inspect the available columns and choose actual column names that exist in the parquet.
- Prefer a mixture like:
  - 3 continuous cost/utilization features from 2017: direct_msk_cost_annual, msk_procedure_quarterly_range_2017, msk_procedure_quarterly_kurtosis_2017
  - 1 binary indicators: has_Long_term_Drug_Therapy



4. Restrict the sampling experiment to majority-class candidate selection
- This notebook is about Stage A control sampling behavior.
- Separate minority and majority rows in the subsample.
- Run the Stage A selection on the majority class only.
- Choose a small deterministic selection size k, for example:
  - otherwise k = min(500, floor(0.1 * n_majority))
- Also create a random majority baseline selection of the same size.
- @experiments_compare_random_vs_curation.py (304-394) is a good start to look at.

6. Build multiple sampling-time warp variants
Important:
These warps are ONLY used to compute the geometry for Stage A selection.
The downstream predictive representation is not the focus here.

Create at least these warp variants on the 2 selected features:

A. raw_simple
- very light preprocessing only
- binary as 0/1
- continuous/count features with simple z-score or min-max scaling
- no aggressive tail control

B. clipped_robust
- binary unchanged
- skewed/count/continuous utilization-cost features clipped at high percentile (for example p95 or p97.5 computed on majority rows)
- then robust-scaled using median/IQR or another bulk-based scale

C. log_clip
- for positive skewed/count features: log1p first
- then clip at high percentile
- then robust-scale
- binary unchanged

D. bounded_rank
- for skewed continuous/count features: map to empirical CDF / percentile rank on majority rows
- optionally clip to [0.02, 0.98] or similar
- re-center if useful
- binary unchanged

7. For each warp, run Stage A selection on majority rows
For each warp variant:
- compute transformed features on the majority subsample only, using statistics from that majority subsample
- run Stage A k-center / farthest-first to select k majority rows
- save selected ENROLID or row index
- also generate a random majority sample baseline of size k

8. Evaluate whether the warp "worked"
This notebook should emphasize diagnostics, not just downstream model metrics.

For each warp-selected majority subset, compare against:
- the full majority subsample
- the random majority subset of the same size
- optionally raw_simple-selected subset

Create diagnostics in three categories.

Category 1: composition of selected controls
Compare selected subsets on:
- 2017 cost distribution
- 2018 cost distribution if present in the data
- utilization summaries
- any obvious intensity/count variables among the 10 selected features
Show:
- histograms / KDEs / boxplots / violin plots
- summary table with mean, median, p90, p95
Goal:
check whether warps reduce enrichment for high-cost/high-utilization tails relative to raw Stage A.

Category 2: density / atypicality of selected controls
Define one or two simple majority-geometry diagnostics in the ORIGINAL lightly normalized feature space:
- distance to majority medoid
- kNN density proxy (e.g. average distance to 10 nearest majority neighbors)
Then compare selected subsets vs random:
- are raw Stage A selections more atypical / lower density?
- do warped Stage A selections move back toward denser bulk regions?
Show:
- summary tables
- histograms
- maybe scatterplot of atypicality vs 2017 cost colored by selection indicator

Category 3: geometric coverage of the majority bulk
I want to see whether the selected set covers multiple bulk submodes instead of just tail points.
Implement one or two simple checks:
- cluster the majority subsample in the original lightly normalized feature space using k-means or another simple method
- compute cluster occupancy / coverage of selected subsets
- report how many clusters are represented and how balanced the selected subsets are across bulk clusters
Also include
- overlay selected points from each warp and from random
The purpose is qualitative: does the warp make Stage A cover bulk modes more sensibly?

10. Notebook structure
Please organize the notebook with markdown headings like:
- Setup
- Load data
- Target and stratified subsample
- Feature selection
- Warp definitions
- Stage A sampling
- Composition diagnostics
- Density / atypicality diagnostics
- Geometry / cluster coverage diagnostics
- Optional downstream sanity check
- Takeaways

11. Implementation details
- Use fixed random seed(s)
- Keep runtime reasonable for notebook use
- Prefer readable Pandas / NumPy / sklearn code
- Save a few compact intermediate objects if useful, but do not overengineer
- Make plots publication-quality enough to inspect, but the notebook is exploratory
- Be explicit about which transformations are fit on which rows

12. Final interpretation cell
End the notebook with a concise markdown summary answering:
- Which warps most reduced high-cost / high-utilization tail enrichment?
- Which warps made Stage A selections look more bulk-dense / less atypical?
- Which warps preserved or improved broad coverage of majority submodes?
- Which 1–2 warp variants seem most promising for scaling up into the real experiment pipeline?

Important:
- Do not silently invent column names; inspect the parquet and choose actual existing columns.
- If ENROLID exists, preserve it in outputs; otherwise preserve row index.
- Keep the notebook self-contained and executable.
- Prefer simple feature warps first; no TF-IDF or SVD unless absolutely necessary.
- The point is to test whether sampling-time warping alone can make deterministic Stage A behave more like representative bulk-majority selection and less like tail/outlier selection.