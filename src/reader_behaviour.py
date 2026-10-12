"""Reader-behaviour analysis for RIVA (Section 5.1 and 5.2 of the manuscript).

Reads annotations.json straight from the dataset archive. Optionally uses the cached
candidate crops and embeddings written by riva_pipeline.py for the image check.

Usage:
    python src/reader_behaviour.py <riva_1.0.zip> [--cache out_dir] [--out results/reader_behaviour.txt]
"""
import argparse, collections, json, pathlib, zipfile
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

READERS = [f"annotator_{i}" for i in range(1, 5)]
LES = {"SCC", "HSIL", "ASCH", "LSIL", "ASCUS", "ASC-H", "ASC-US"}
RADII = (1.5, 1.85, 2.0)
CALIBRATED = 1.85

ap = argparse.ArgumentParser()
ap.add_argument("zip"); ap.add_argument("--cache", default=None); ap.add_argument("--out", default=None)
args = ap.parse_args()
lines = []
def emit(s=""): print(s, flush=True); lines.append(s)

zf = zipfile.ZipFile(args.zip)
A = json.loads(zf.read([n for n in zf.namelist() if n.endswith("annotations.json")][0]))
multi = {k: v for k, v in A.items() if len(v) == 4}
emit(f"patches: {len(A)}  four-reader: {len(multi)}  single-reader: {len(A) - len(multi)}")
emit(f"points: {sum(len(l) for p in A.values() for l in p.values())}")
single = collections.Counter(next(iter(v)) for k, v in A.items() if len(v) == 1)
emit("single-reader patches per reader: " + ", ".join(f"{r}={single[r]}" for r in READERS))

def groups(patch, r):
    pts, who, lbl = [], [], []
    for a, lst in patch.items():
        for d in lst:
            pts.append((d["x"], d["y"])); who.append(a); lbl.append(d["keypointlabels"])
    if not pts: return []
    lab = DBSCAN(eps=r, min_samples=1).fit_predict(np.array(pts))
    g = collections.defaultdict(list)
    for i, l in enumerate(lab): g[l].append(i)
    out = []
    for idx in g.values():
        labs = collections.Counter(lbl[i] for i in idx)
        out.append(dict(who=[who[i] for i in idx], labels=labs, top=labs.most_common(1)[0][0]))
    return out

def cls_of(top):
    return "lesion" if top in LES else ("INFL" if top.startswith("INF") else "other-nonlesion")

for r in RADII:
    emit(f"\n================ radius {r}% of image size ================")
    rows = []
    for name, p in multi.items():
        gs = groups(p, r); dense = len(gs) >= 25
        for g in gs: rows.append((g, dense))
    n = len(rows)
    merged = sum(len(g["who"]) != len(set(g["who"])) for g, _ in rows)
    dist = collections.Counter(len(set(g["who"])) for g, _ in rows)
    emit(f"matched cells: {n}   merged groups: {merged} ({100*merged/n:.1f}%)")
    emit("readers per cell: " + ", ".join(f"{k}: {v} ({100*v/n:.0f}%)" for k, v in sorted(dist.items())))
    f1, f2 = dist[1], dist[2]
    chao = n + (3 / 4) * f1 * f1 / (2 * f2)
    emit(f"Chao2 estimate of all cells: {chao:.0f}  (unseen about {chao - n:.0f})")
    emit("share of cells marked by someone else that the reader also marked:")
    emit(f"  {'reader':12s} {'all':>5s} {'lesion':>7s} {'INFL':>6s} {'other':>6s} {'dense':>6s} {'sparse':>7s}")
    for a in READERS:
        hit = collections.defaultdict(lambda: [0, 0])
        for g, dense in rows:
            who = set(g["who"])
            if not (who - {a}): continue
            h = a in who
            for key in ("all", cls_of(g["top"]), "dense" if dense else "sparse"):
                hit[key][0] += h; hit[key][1] += 1
        f = lambda k: 100 * hit[k][0] / max(hit[k][1], 1)
        emit(f"  {a:12s} {f('all'):4.0f}% {f('lesion'):6.0f}% {f('INFL'):5.0f}% {f('other-nonlesion'):5.0f}% {f('dense'):5.0f}% {f('sparse'):6.0f}%")
    skipped = collections.defaultdict(collections.Counter)
    for g, _ in rows:
        who = set(g["who"])
        if len(who) < 2: continue
        n_les = sum(c for l, c in g["labels"].items() if l in LES)
        for a in READERS:
            if a in who or len(who - {a}) < 2: continue
            skipped[a]["lesion-called" if n_les * 2 >= len(g["who"]) else "non-lesion-called"] += 1
    emit("cells skipped although at least two other readers marked them:")
    for a in READERS:
        emit(f"  {a}: {sum(skipped[a].values()):5d}  {dict(skipped[a])}")
    if r == CALIBRATED:
        far = {a: [] for a in READERS}
        for name, p in multi.items():
            pts = {a: np.array([(d["x"], d["y"]) for d in lst]) for a, lst in p.items() if lst}
            for a, P in pts.items():
                oth = [Q for b, Q in pts.items() if b != a]
                if not oth: continue
                d, _ = NearestNeighbors(n_neighbors=1).fit(np.vstack(oth)).kneighbors(P)
                far[a] += list(d[:, 0])
        emit(f"marks with no other-reader mark within {r}% / between {r}% and 4%:")
        for a in READERS:
            d = np.array(far[a])
            emit(f"  {a}: {100*np.mean(d > r):.0f}% / {100*np.mean((d > r) & (d <= 4)):.0f}%")

# ------------------------------------------------------------------ image check
if args.cache:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    c = pathlib.Path(args.cache)
    Z = np.load(c / "candidates.npz"); E = np.load(c / "emb_resnet18.npy")
    M, BG, PAT = Z["M"], Z["BG"].astype(bool), Z["PAT"]
    n_read = M.sum(1)
    kind = np.where(BG, "background", np.where(n_read >= 3, "consensus", np.where(n_read == 1, "solo", "other")))
    rd = np.where(n_read == 1, M.argmax(1), -1)
    lab = (kind == "consensus").astype(int)
    train = np.where((kind == "consensus") | (kind == "background"))[0]
    solo = np.where(kind == "solo")[0]
    score = np.full(len(kind), np.nan)
    for tr, te in GroupKFold(5).split(E[train], lab[train], PAT[train]):
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(E[train[tr]], lab[train[tr]])
        score[train[te]] = clf.predict_proba(E[train[te]])[:, 1]
        held = set(PAT[train[te]]); m = np.array([p in held for p in PAT[solo]])
        score[solo[m]] = clf.predict_proba(E[solo[m]])[:, 1]
    thr = np.nanpercentile(score[kind == "consensus"], 10)
    emit("\n================ image check (frozen ResNet18, 64 px crops, patch-grouped 5-fold) ================")
    emit(f"threshold = 10th percentile of consensus scores = {thr:.2f}")
    emit(f"median score: consensus {np.nanmedian(score[kind=='consensus']):.2f}, background {np.nanmedian(score[kind=='background']):.2f}; "
         f"background above threshold {100*np.nanmean(score[kind=='background'] > thr):.0f}%")
    for i, a in enumerate(READERS):
        m = (kind == "solo") & (rd == i)
        emit(f"  {a}: solo marks {m.sum():4d}  share cell-like {100*np.nanmean(score[m] > thr):.0f}%")

if args.out:
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n")
