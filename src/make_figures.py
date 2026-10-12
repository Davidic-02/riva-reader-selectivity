"""Figures for the manuscript, built from the text files in results/.

Usage: python src/make_figures.py [results_dir] [figures_dir]
Needs: matplotlib. Reads results_resnet18.txt, results_dino.txt and reader_behaviour.txt.
"""
import pathlib, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
READERS = ["annotator_1", "annotator_2", "annotator_3", "annotator_4"]

# ---- Figure 1: reader marking rates by class group (radius 1.85 block of reader_behaviour.txt)
txt = (RES / "reader_behaviour.txt").read_text()
block = txt.split("radius 1.85% of image size")[1].split("cells skipped")[0]
rows = {}
for m in re.finditer(r"(annotator_\d)\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%", block):
    rows[m.group(1)] = [int(x) for x in m.groups()[1:]]
cols = ["All cells", "Lesion", "Inflammation", "Other non-lesion"]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
w = 0.2
for j, name in enumerate(cols):
    ax.bar(np.arange(4) + (j - 1.5) * w, [rows[r][j] for r in READERS], w, label=name)
ax.set_xticks(range(4)); ax.set_xticklabels([r.replace("annotator_", "Reader ") for r in READERS])
ax.set_ylabel("Cells also marked (%)"); ax.set_ylim(0, 100); ax.legend(frameon=False, ncol=2)
ax.set_title("Share of cells marked by someone else that each reader also marked")
fig.tight_layout(); fig.savefig(OUT / "fig1_reader_rates.png", dpi=300); plt.close(fig)

# ---- Figure 2: correction effect on lesion average precision (reference >=2 readers)
def corr(path):
    t = path.read_text().split("reference = marked by >=3")[0]
    out = {}
    for m in re.finditer(r"(\w+) - naive \(lesion\):.*?AP ([+-]\d\.\d+) \[([+-]\d\.\d+),([+-]\d\.\d+)\]", t):
        out[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
    return out
c = {"ResNet18": corr(RES / "results_resnet18.txt"), "DINOv2-small": corr(RES / "results_dino.txt")}
labels = {"pu_pred": "Predicted class + density proxy", "pu_reader": "Per-reader rate only", "pu_const": "Constant rate"}
fig, ax = plt.subplots(figsize=(6.2, 2.8))
for k, (feat, mark) in enumerate(c.items()):
    for i, key in enumerate(["pu_pred", "pu_reader", "pu_const"]):
        m, lo, hi = mark[key]; y = i + (k - 0.5) * 0.25
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=f"C{k}", capsize=2, label=feat if i == 0 else None)
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(3)); ax.set_yticklabels([labels[k] for k in ["pu_pred", "pu_reader", "pu_const"]])
ax.set_xlabel("Change in lesion average precision vs naive (95% interval)", fontsize=8)
ax.invert_yaxis(); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig2_correction.png", dpi=300); plt.close(fig)

# ---- Figure 3: deferral risk-coverage
def deferral(path):
    t = path.read_text().split("=== deferral")[1]
    covs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]; out = {}
    for name in ["confidence", "predicted disagreement", "both (mean rank)", "random"]:
        m = re.search(re.escape(name) + r"\s+((?:\d\.\d+\s*){6})", t)
        out[name] = [float(x) for x in m.group(1).split()]
    return covs, out
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
for ax, (feat, f) in zip(axes, [("ResNet18", "results_resnet18.txt"), ("DINOv2-small", "results_dino.txt")]):
    covs, d = deferral(RES / f)
    for k, v in d.items(): ax.plot(covs, v, marker="o", ms=3, label=k)
    ax.set_title(feat); ax.set_xlabel("Coverage")
axes[0].set_ylabel("Error among accepted cells"); axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig3_deferral.png", dpi=300); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))
