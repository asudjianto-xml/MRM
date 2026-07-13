# GBDT as a Kernel: Finding Where Your Model Is Weak

**Track:** Tabular · **Status:** draft

A gradient-boosted tree ensemble quietly defines a *kernel*: two points are
similar if they land in the same leaves across the trees. This article teaches
that idea and then puts it to work — using the GBDT's own kernel to partition
the data (via a Nyström spectral embedding) and reveal the regions where the
model underperforms.

## Arc

1. **Fit a GBDT.** Train an XGBoost model on a tabular dataset and look at its
   overall performance (`TestSuite.diagnose_accuracy_table`).
2. **GBDT as a kernel.** The leaf co-membership matrix *is* a kernel `K`. In
   Modeva this is the FuseKernel with the XGBoost channel only
   (`use_xgb=True, use_rbf=False, use_spectral=False`) — a pure leaf
   co-membership kernel. (`MoGBDTKernelRegressor.get_neighbor_analysis` makes
   "who shares your leaves" explicit.)
3. **Nyström spectral clustering.** `K` is `n x n`; Nyström picks landmarks and
   extends the normalized (Ng–Jordan–Weiss) eigenvectors, so the eigensolve is
   linear in `n`. Cluster the data in that spectral embedding.
4. **Per-cluster performance = weakness map.** Report the model metric per
   cluster on train and test. Low headline metric or a large train/test gap
   flags a region the GBDT models poorly.

## Key Modeva API

```python
from modeva import DataSet, TestSuite
from modeva.models import MoFuseKernelRegressor

fk = MoFuseKernelRegressor(backend="xgboost",
                           use_xgb=True, use_rbf=False, use_spectral=False,
                           solver="nystrom", random_state=0)
fk.fit(ds.train_x, ds.train_y.ravel())          # pure GBDT leaf co-membership kernel

res = fk.diagnose_weak_clusters(ds, n_clusters=5)   # Nyström spectral clustering of that kernel
res.table                      # per-cluster train/test metrics (+ ALL row)
res.value["worst_clusters"]    # weakest-cluster ranking
res.plot()                     # per-cluster metric bar chart
```

## Contents

- `notebooks/` — the runnable companion notebook.
