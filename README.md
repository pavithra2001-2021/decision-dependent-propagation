# Decision-dependent propagation of correlated feature-estimation error: research code

This repository accompanies the manuscript *Decision-dependent propagation of correlated feature-estimation error in
connectome-based machine learning*. It contains the analysis code, the settings it uses, an environment
specification, the scripts that regenerate every number, table and figure of the manuscript and its supplement from derived results,
and the derived per-participant summary tables. Raw imaging data and preprocessed regional time series are not included; they are
regenerated from the public sources below.

## Layout
| Path | Content |
|---|---|
| `src/p6/` | Analysis library: data access and preprocessing orchestration for CoRR NYU_2 (`corr_nyu2.py`, `run_corr_dparsf.sh`, `dparsf_run.m`, `qc_corr.m`); the confirmation analysis (`exp_confirm.py`); the discovery analysis and edge-level estimators (`exp_retest.py`); decision-level propagation, J-CBB and uHAC (`exp_mechanism.py`, `uncertainty.py`, `fc.py`); statistics (`stats.py`, `metrics.py`); value extraction from result files (`manuscript_assets.py`) |
| `configs/analysis_settings.json` | Fixed settings (seeds, replicates, thresholds, method lists), read from the module constants of the code in this archive |
| `results/` | Derived results of the discovery (TCP) and confirmation (NYU_2) studies |
| `data/external/derived/corr_nyu2/qc.json` | Quality-control summary of the NYU_2 preprocessing |
| `analysis/01_analyses/` | Additional analyses: one folder per analysis with its scripts and derived results; `common.py` holds the shared loaders; `external_metadata/` holds the public NYU_2 session files (age, sex) |
| `analysis/02_manuscript/` | `build_values.py`, `build_supp_assets.py` and `build_values_extra.py` turn the result files into the numerical macros (`values.tex`, `supp_values.tex`, `values_extra.tex`) and table bodies; `macro_provenance.csv` maps every macro to its source file and field |
| `analysis/03_figures/` | `build_figures.py` and `build_figures_extra.py` (all data figures), `Figure_1.tex` and `graphical_abstract.tex` (TikZ) |
| `requirements.txt` | Python version and package versions used |

## Regenerating the numbers, tables and figures (no imaging data needed)
```
cd analysis/02_manuscript
python build_values.py        # writes values.tex, tab_sim.tex, tab_b1.tex and starts macro_provenance.csv
python build_supp_assets.py   # writes supp_values.tex and supp_hyper.tex
python build_values_extra.py     # writes values_extra.tex and tab_len_full.tex
cd ../03_figures
python build_figures_extra.py    # writes Figure_2 ... Figure_8 and Figure_S1 ... Figure_S5
```
Run the three value scripts in this order. Running these steps on the files in this repository
reproduces the values and figures of the manuscript exactly.

## Re-running the analyses from the public data
1. **CoRR NYU_2 (confirmation cohort).** `PYTHONPATH=src python -m p6.corr_nyu2 full --workers 2` downloads session-1 rest runs and T1
   images from the INDI bucket, preprocesses each run with DPARSF (DPABI V10.0, GNU Octave 8.4.0, SPM12 r7771) under WSL via
   `src/p6/run_corr_dparsf.sh`, and runs the quality checks. Set `P6_ROOT` to the WSL path of this folder, and replace `/path/to/project`
   in `src/p6/corr_nyu2.py` and in `analysis/01_analyses/second_session/run_preprocessing.py` with the same path.
2. **Confirmation analysis.** `PYTHONPATH=src python -m p6.exp_confirm --confirm CONFIRMATORY-ANALYSIS` computes the per-participant
   table `results/confirmation_participants.csv.gz` and `results/confirmation_results.json`.
3. **TCP (discovery cohort).** Place the provider-parcellated rest series of OpenNeuro ds005237 v1.1.3 under
   `data/external/raw/ds005237/`; the functions in `src/p6/exp_retest.py` (`cohort`, `features`, `analyse`) compute the discovery results.
4. **Additional analyses.** Run the scripts in each folder of `analysis/01_analyses/` (for example `python run.py`); each writes its
   `results/` folder. The second session of NYU_2 is preprocessed with `second_session/run_preprocessing.py` and analysed with `analyse.py`.

## uHAC reference implementation (NumPy)
A self-contained version of Algorithm 1 for a linear decision (or the gradient of a differentiable one) on Fisher-z correlations.
`src/p6/exp_mechanism.py` (`projected_influence_fast`, `lrv`, `decision_sigma`) is the implementation used in the analyses.
```python
import numpy as np


def uhac_sd(X, w, b):
    # Decision SD from one run. X: (T, N) array; w: (E,) weights over channel pairs in
    # np.triu_indices(N, 1) order; b: block length (from the bootstrap, or an automatic bandwidth rule).
    T, N = X.shape
    Y = (X - X.mean(0)) / X.std(0)                       # 1: standardise each channel
    R = np.clip(Y.T @ Y / T, -0.999999, 0.999999)        # 2: sample correlations, clipped off the diagonal
    np.fill_diagonal(R, 1.0)
    i, j = np.triu_indices(N, 1)
    G = np.zeros((N, N))
    G[i, j] = w / (1.0 - R[i, j] ** 2)                   # 3: G_ij = w_e / (1 - R_ij^2), zero diagonal
    G = G + G.T
    c = 0.5 * (G * R).sum(1)                             # 4: c_i = (1/2) sum_j G_ij R_ij
    u = 0.5 * np.sum((Y @ G) * Y, axis=1) - (Y ** 2) @ c # 5: projected influence series u_t
    u = u - u.mean()                                     # 6: centre; Bartlett bandwidth b_T = 2b
    bT = 2.0 * b
    H = min(int(np.ceil(bT)), T - 1)
    V = u @ u / T                                        # 7-8: Bartlett-weighted long-run variance
    for h in range(1, H + 1):
        V += 2.0 * max(0.0, 1.0 - h / bT) * (u[:-h] @ u[h:]) / T
    return np.sqrt(V / T)                                # 9: decision SD
```

## Derived per-participant summary tables
- `results/confirmation_participants.csv.gz`: NYU_2, 158 participants x 3 prespecified decisions; decision values of both runs, run-wise decision
  SDs for each propagation method, and the NLPD of each forecast.
- `results/discovery_decisions_AP.csv.gz`, `results/discovery_decisions_PA.csv.gz`: TCP, per-participant decision values and SDs.
- `analysis/01_analyses/*/results/participants_*.csv(.gz)` and `*.json`: per-participant and summary results of the additional analyses.

## Data sources and licences
- **Transdiagnostic Connectome Project**, OpenNeuro ds005237 version 1.1.3 (https://doi.org/10.18112/openneuro.ds005237.v1.1.3),
  released under CC0 1.0, which places no restriction on redistributing derived data. Cite Chopra et al. (2025), Scientific Data 12, 923.
- **CoRR NYU_2**, International Neuroimaging Data-sharing Initiative (https://doi.org/10.15387/fcp_indi.corr.nyu2), imaging data released
  under the Open Data Commons Attribution License (ODC-By) v1.0. ODC-By permits sharing and adapting the data, including derived databases,
  provided the source is attributed. The derived tables and session files included here are shared on that basis; cite Zuo et al. (2014),
  Scientific Data 1, 140049, and the dataset DOI.

Participant identifiers are the public identifiers of the source datasets.

## Licence
Public access is provided with copyright retained; no open-source reuse licence is granted in this version. The derived tables remain subject to the attribution terms of the source data listed above.
