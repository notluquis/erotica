# Quickstart

A minimal membership run on a Gaia field, from an `astropy` table of sources.

```python
from astropy.table import QTable
from erotica.core.clustering import Clustering

data = QTable.read("my_gaia_field.ecsv")   # ra, dec, pmra, pmdec, parallax, ...

clu = Clustering(data)

# Pseudo-probability membership: sweep min_cluster_size, score by
# recovery frequency x within-cluster strength. Default feature space is
# 2D proper motion — see the membership guide for the alternatives.
clu.search_pseudoprobability(columns=("pmra", "pmdec"))

clu.clustering_statistics()                  # counts: members, outliers, clusters
summary = clu.get_cluster_summary()          # pandas DataFrame, one row per cluster
clu.save_results("members.ecsv", format="ascii.ecsv")
```

```{note}
`show_results()` reports the hyper-parameter search and only works after
{meth}`~erotica.core.Clustering.search` (the grid/Optuna path). After
`search_pseudoprobability` it will just tell you to run `.search()` first — use
`clustering_statistics()` and `get_cluster_summary()` instead, as above.
```

The `columns` argument **is** the feature-space choice. Clustering on velocity space
(recommended when parallaxes are informative) is one edit:

```python
clu.search_pseudoprobability(columns=("pmra", "pmdec", "parallax"))
```

**But standardize first when you mix units:** EROTICA clusters on the raw column values and
HDBSCAN uses a Euclidean metric, so raw `pmra,pmdec,parallax` under-weights parallax (and
raw 5D is dominated by sky position). Rescale each axis to a common spread before passing
them — see the [membership guide](guides/membership.md).

Which columns to cluster on, whether to fold in measurement errors, and whether to treat
membership as a number or a posterior are all deliberate decisions with trade-offs — the
[membership guide](guides/membership.md) lays them out.
