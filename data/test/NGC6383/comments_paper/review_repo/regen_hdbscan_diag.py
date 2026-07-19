#!/usr/bin/env python3
"""Standalone regeneration of the HDBSCAN diagnostic figures, replacing the
notebook versions: min_cluster_size.pdf (Fig. B.1) and
condensed_cluster_tree_NGC6383.pdf (Fig. B.2), with the common figure styling
shared by the other regen_*.py scripts.

B.1 plots cluster size vs min_cluster_size only for sweep runs passing the
branch-stability criteria stated in the caption (condensed-tree lambda_max >= 8
and recovered branch size in 200-701), from the audited sweep track CSV.
B.2 is drawn from the saved production Clustering object (40 arcmin dill)."""
import sys
sys.path.insert(0, "/Users/notluquis/COSMIC")
import dill, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt

B = "/Users/notluquis/COSMIC/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"
clu = dill.load(open(f"{B}/comments_paper/radius_robustness/generated/dill/ngc6383_40_paperfaithful.dill", "rb"))

# ---- Fig. B.1: min_cluster_size sweep, from the clusterer's own delivered data ----
# clu.pseudoprobability_results_ holds, per candidate min_cluster_size, the branch
# size and the HDBSCAN condensed-tree lambda (lambda_val.max from condensed_tree_).
# We show the runs passing the caption's branch-stability criteria (lambda_max >= 8
# and recovered branch size 200-701); the optimum is the clusterer's own selection.
df = pd.DataFrame([{"mcs": r["min_cluster_size"], "size": r["desired_len"], "lam": r["lambda_value"]}
                   for r in clu.pseudoprobability_results_])
f = df[(df.lam >= 8) & (df["size"].between(200, 701))].sort_values("mcs")
best_mcs = int(clu.pseudoprobability_selected_["min_cluster_size"])   # clusterer-selected (43)
max_size = int(f["size"].max())                                       # 701
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f.mcs, f["size"], color="steelblue")
ax.axvline(best_mcs, color="steelblue", ls="--", label=f"Optimal Min Cluster Size: {best_mcs}")
ax.axhline(max_size, color="olivedrab", ls="--", label=f"Max Cluster Size: {max_size}")
ax.set_xlabel("Min Cluster Size"); ax.set_ylabel("Cluster Size"); ax.legend()
fig.tight_layout(); fig.savefig(FIG + "min_cluster_size.pdf", bbox_inches="tight"); plt.close(fig)
print(f"wrote min_cluster_size.pdf  (mcs {f.mcs.min()}-{f.mcs.max()}, peak {max_size} at {best_mcs})")

# ---- Fig. B.2: condensed cluster tree -- HDBSCAN's native condensed_tree_.plot() ----
clu.plot_condensed_tree(save_path=FIG + "condensed_cluster_tree_NGC6383.pdf")
plt.close("all")
print("wrote condensed_cluster_tree_NGC6383.pdf")
