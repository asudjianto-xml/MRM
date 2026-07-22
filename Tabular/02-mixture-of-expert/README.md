# Mixture of Experts: Repairing Where a Monotone Credit Model Is Weak

**Track:** Tabular · **Status:** draft

Article 01 used a GBDT's own kernel to find the regions where a model is weak. This
article continues the story on a **credit default** problem: fit an **interpretable,
monotone** base model (CatBoost, depth-2 trees), use its kernel to locate the weak
clusters, then **repair** them with a **Mixture of Experts** — without giving up the
model's economic (monotonicity) guarantees.

## Arc

1. **Interpretable, monotone base model.** A depth-2 CatBoost (main effects + pairwise
   interactions) with per-feature **monotonicity constraints** in the economically
   correct direction — tuned by 5-fold CV. Its **FANOVA** decomposition (effect and
   feature importance, monotone main-effect and interaction shapes) is read straight from
   the fitted model — inherent interpretability, not a post-hoc surrogate.
2. **CatBoost as a kernel.** The leaf co-membership matrix is a kernel `K`; in Modeva
   this is the FuseKernel with the tree channel only (`backend="catboost"`,
   `use_rbf=False, use_spectral=False`), carrying the same tuned monotone params.
3. **Nyström spectral clustering → weakness map.** `diagnose_weak_clusters` clusters the
   applicants in a Nyström spectral embedding of `K` and reports per-cluster train/test
   metrics. Characterize the weak region with **PSI** (`data_drift_test`).
4. **Repair with a Mixture of Experts.** `MoMoEClassifier` with monotone CatBoost
   depth-2 experts routes each region to a specialist. The per-cluster AUC lift lands on
   exactly the clusters the kernel flagged as weak; the strong clusters are untouched.

## Key Modeva API

```python
from modeva import DataSet, TestSuite
from modeva.models import (MoCatBoostClassifier, MoFuseKernelClassifier,
                           ModelTuneGridSearch, MoMoEClassifier)

# monotone_constraints must be a positional STRING (a list/dict breaks sklearn clone,
# a tuple is rejected by catboost.fit — only the string survives both HPO and MoMoE).
mono = "(" + ",".join(str(direction[f]) for f in ds.feature_names) + ")"

# interpretable, monotone base model
cb = MoCatBoostClassifier(depth=2, monotone_constraints=mono, verbose=0, **best_params)
cb.fit(ds.train_x, ds.train_y.ravel())

# inherent interpretability (FANOVA) — all MoCharts
ts = TestSuite(ds, cb)
ts.interpret_ei().plot()                          # effect importance
ts.interpret_fi().plot()                          # feature importance
ts.interpret_effects(features="score").plot()     # monotone main-effect shape
ts.interpret_effects(features=("dti", "score")).plot()   # pairwise interaction

# same model, as a pure leaf co-membership kernel
fk = MoFuseKernelClassifier(backend="catboost",
                            use_xgb=True, use_rbf=False, use_spectral=False,
                            solver="nystrom",
                            gbdt_params={**best_params, "depth": 2,
                                         "monotone_constraints": mono})
fk.fit(ds.train_x, ds.train_y.ravel())
weak = fk.diagnose_weak_clusters(ds, n_clusters=5)   # weakness map

# repair: mixture of monotone catboost experts
moe = MoMoEClassifier(n_clusters=5, expert="catboost",
                      depth=2, monotone_constraints=mono, verbose=0)
moe.fit(ds.train_x, ds.train_y.ravel())

# MoE stays interpretable + monotone after repair
ts_moe = TestSuite(ds, moe)
ts_moe.interpret_moe_cluster_analysis().plot()            # per-expert region + AUC
ts_moe.interpret_effects_moe_average(features="dti").plot()   # gate-averaged effect, still monotone
```

## Contents

- `notebooks/mixture_of_experts_repair.ipynb` — the runnable companion notebook
  (loads local `credit_default.csv`).
