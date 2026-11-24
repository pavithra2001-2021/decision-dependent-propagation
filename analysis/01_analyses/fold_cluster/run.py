"""Fold-cluster bootstrap interval for the primary endpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

C = K.nyu2_cohort()
y = (C.sex == "female").astype(int).values
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
D1 = P[P.D == "D1"].reset_index(drop=True)
assert list(D1.pid) == list(C.pid), "participant order differs from the cohort order"
diff = (D1.nlpd_J_cbb - D1.nlpd_I_xdf).values
folds = [te for _, te in K.nyu2_outer_folds(y)]

fold_med = np.array([np.median(diff[te]) for te in folds])
assert round(fold_med.min(), 2) == -11.28 and round(fold_med.max(), 2) == -1.35 and (fold_med < 0).all(), fold_med

rng = np.random.default_rng(20261024)
B = 10000
meds = np.empty(B)
for b in range(B):
    pick = rng.integers(0, len(folds), len(folds))
    meds[b] = np.median(np.concatenate([diff[folds[k]][rng.integers(0, len(folds[k]), len(folds[k]))] for k in pick]))
lo, hi = np.percentile(meds, [2.5, 97.5])
S = {"n": int(len(diff)), "n_folds": len(folds), "median_diff": float(np.median(diff)),
     "fold_cluster_ci95": [float(lo), float(hi)], "resamples": B, "seed": 20261024,
     "fold_medians": [float(x) for x in fold_med]}
(OUT / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
print(json.dumps(S, indent=1))
