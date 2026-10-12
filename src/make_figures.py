"""Figures for the manuscript, built from the text files in results/.

Usage: python src/make_figures.py [results_dir] [figures_dir]
Needs: matplotlib. Reads results_resnet18.txt, results_dino.txt, reader_behaviour.txt and mixture_model.txt.
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

# ---- Figure 3: reader marking rates by class group (radius 1.85 block of reader_behaviour.txt)
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
fig.tight_layout(); fig.savefig(OUT / "fig3_reader_rates.png", dpi=300); plt.close(fig)


from matplotlib.patches import FancyBboxPatch

# ---- Figure 1: study design (schematic)
fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
def box(x, y, w, h, t, c="#dbe9f6"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="#33475b", lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
def arrow(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#33475b"))
box(0.1, 1.9, 1.9, 1.3, "RIVA\n959 patches\n386 read by 4 readers\n573 by one reader")
box(2.5, 3.4, 2.3, 1.1, "A. Reader behaviour\nmatch marks across readers;\nrates, skipped cells,\nimage check, mixture", "#fde9c9")
box(2.5, 0.4, 2.3, 1.3, "B. Correction experiment\n10,379 candidates, frozen\nResNet18 / DINOv2-small;\nsimulate one reader per patch", "#d9efd9")
box(5.4, 0.4, 2.0, 1.3, "Train: naive, PU const,\nPU reader, PU predicted,\nPU true*, oracle", "#d9efd9")
box(8.0, 0.4, 1.9, 1.3, "Test vs reference\n(>=2 or >=3 readers);\nslide bootstrap", "#d9efd9")
box(5.4, 3.4, 4.5, 1.1, "Results: rates differ by reader and class;\nunmarked cells are a mixture;\nclass-aware correction not distinguishable from zero", "#fde9c9")
box(8.0, 2.0, 1.9, 1.0, "C. Deferral:\nconfidence vs predicted\nreader share", "#e7dff3")
arrow(2.0, 2.7, 2.5, 3.8); arrow(2.0, 2.3, 2.5, 1.2); arrow(4.8, 1.05, 5.4, 1.05); arrow(7.4, 1.05, 8.0, 1.05)
arrow(4.8, 3.95, 5.4, 3.95); arrow(8.95, 1.7, 8.95, 2.0)
fig.tight_layout(); fig.savefig(OUT / "fig1_design.png", dpi=300); plt.close(fig)

# ---- Figure 2: readers per cell (radius 1.85) with Chao2 estimate
dist = re.search(r"readers per cell: 1: (\d+) .*?2: (\d+) .*?3: (\d+) .*?4: (\d+) ", block).groups()
dist = [int(v) for v in dist]
chao = re.search(r"Chao2 estimate of all cells: (\d+)\s+\(unseen about (\d+)\)", block).groups()
fig, ax = plt.subplots(figsize=(4.6, 3.0))
bars = ax.bar(["1", "2", "3", "4"], dist, color=["#c9d6e8", "#9fb7d6", "#6f97c4", "#3f74ad"])
for b, v in zip(bars, dist): ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}", ha="center", fontsize=8)
ax.set_xlabel("Readers who marked the cell"); ax.set_ylabel("Matched cells")
ax.set_title(f"7,470 matched cells; Chao2 estimate of unseen cells about {int(chao[1]):,}", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig2_readers_per_cell.png", dpi=300); plt.close(fig)

# ---- Figure 4: skipped cells and image check
skip_block = txt.split("radius 1.85% of image size")[1].split("cells skipped although at least two other readers marked them:")[1].split("marks with no")[0]
sk = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"(annotator_\d):\s+\d+\s+\{'non-lesion-called': (\d+), 'lesion-called': (\d+)\}", skip_block)}
chk = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"(annotator_\d): solo marks\s+(\d+)\s+share cell-like (\d+)%", txt)}
bg = int(re.search(r"background above threshold (\d+)%", txt).group(1))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
x = np.arange(4); nl = [sk[r][0] for r in READERS]; le = [sk[r][1] for r in READERS]
a1.bar(x, nl, label="called non-lesion", color="#9fb7d6"); a1.bar(x, le, bottom=nl, label="called lesion", color="#c0504d")
a1.set_xticks(x); a1.set_xticklabels([f"R{i+1}" for i in range(4)]); a1.set_ylabel("Cells skipped"); a1.legend(frameon=False, fontsize=7)
a1.set_title("Cells skipped though >=2 others marked them", fontsize=8)
vals = [chk[r][1] for r in READERS]
a2.bar(x, vals, color="#6f97c4", label="solo marks"); a2.axhline(bg, color="k", ls="--", lw=0.8); a2.text(3.45, bg + 2, f"background {bg}%", ha="right", fontsize=7)
for i, r in enumerate(READERS): a2.text(i, vals[i] + 2, f"n={chk[r][0]}", ha="center", fontsize=7)
a2.set_xticks(x); a2.set_xticklabels([f"R{i+1}" for i in range(4)]); a2.set_ylim(0, 100); a2.set_ylabel("Scored cell-like (%)")
a2.set_title("Solo marks, image check", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig4_skipped_imagecheck.png", dpi=300); plt.close(fig)

# ---- Figure 5: mixture model
mx = (RES / "mixture_model.txt").read_text()
pr = [[float(v) for v in m.groups()] for m in re.finditer(r"annotator_\d: ([\d.]+) \| ([\d.]+) \| ([\d.]+)\s+([\d.]+)", mx)]
ev = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in re.finditer(r"^  (\d): +(\d+) vs +([\d.]+)", mx, flags=re.M)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
w = 0.2
for j, name in enumerate(["non-lesion, sparse", "lesion, sparse", "lesion, dense"]):
    a1.bar(x + (j - 1) * w, [pr[i][j] for i in range(4)], w, label=name)
a1.plot(x, [pr[i][3] for i in range(4)], "k^", label="spurious mark")
a1.set_xticks(x); a1.set_xticklabels([f"R{i+1}" for i in range(4)]); a1.set_ylim(0, 1.45); a1.set_ylabel("Probability reader marks cell")
a1.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center"); a1.set_title("Fitted mark probabilities", fontsize=8)
ks = [e[0] for e in ev]; a2.bar(np.array(ks) - 0.18, [e[1] for e in ev], 0.36, label="observed"); a2.bar(np.array(ks) + 0.18, [e[2] for e in ev], 0.36, label="model")
a2.set_xticks(ks); a2.set_xlabel("Readers per cell"); a2.set_ylabel("Cells"); a2.legend(frameon=False, fontsize=7)
a2.set_title("Model fit: readers per cell", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig5_mixture.png", dpi=300); plt.close(fig)

# ---- Figure 6: lesion AP by method (reference >=2 readers)
def lesap(path):
    t = path.read_text().split("reference = marked by >=3")[0]; out = {}
    for line in t.splitlines():
        m = re.match(r"(naive|pu_const|pu_reader|pu_pred|pu_true\*|oracle)\s+[\d.]+\s+[\d.]+ \|\s+[\d.]+\s+([\d.]+)", line)
        if m: out[m.group(1)] = float(m.group(2))
    return out
meths = ["naive", "pu_const", "pu_reader", "pu_pred", "pu_true*", "oracle"]
mlab = ["naive", "PU constant", "PU reader", "PU predicted", "PU true*", "oracle"]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for k, (feat, path) in enumerate([("ResNet18", "results_resnet18.txt"), ("DINOv2-small", "results_dino.txt")]):
    d = lesap(RES / path); ax.bar(np.arange(6) + (k - 0.5) * 0.38, [d[m] for m in meths], 0.38, label=feat)
ax.set_xticks(range(6)); ax.set_xticklabels(mlab, fontsize=8); ax.set_ylim(0.5, 0.7); ax.set_ylabel("Lesion average precision")
ax.legend(frameon=False); ax.set_title("Lesion AP by training method (reference: >=2 readers)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig6_ap_methods.png", dpi=300); plt.close(fig)

# ---- Figure 7: correction effect on lesion average precision (reference >=2 readers)
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
ax.invert_yaxis(); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig7_correction.png", dpi=300); plt.close(fig)

# ---- Figure 8: deferral risk-coverage
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
fig.tight_layout(); fig.savefig(OUT / "fig8_deferral.png", dpi=300); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))
