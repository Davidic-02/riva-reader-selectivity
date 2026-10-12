"""RIVA reader-aware correction experiment, headless version of Colab cells 7-19.

Reads the zip in place (annotations.json + PNGs); nothing is extracted.
Usage: python -I riva_pipeline.py <riva_1.0.zip> <out_dir> [--features resnet18|dino]
"""
import collections, io, json, re, sys, time, zipfile, pathlib
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import rankdata
import torch, torchvision

ZIP, OUT = sys.argv[1], pathlib.Path(sys.argv[2])
FEATS = sys.argv[sys.argv.index("--features") + 1] if "--features" in sys.argv else "resnet18"
OUT.mkdir(parents=True, exist_ok=True)
READERS = [f"annotator_{i}" for i in range(1, 5)]
LES = {"SCC", "HSIL", "ASCH", "LSIL", "ASCUS", "ASC-H", "ASC-US"}
CROP, RADIUS = 64, 1.85
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
torch.set_num_threads(max(1, torch.get_num_threads()))

# ------------------------------------------------------------------ candidates
cache = OUT / "candidates.npz"
if cache.exists():
    Z = np.load(cache, allow_pickle=False)
    crops, M, BG, LSN, DEN, PAT, GRP = (Z[k] for k in ("crops", "M", "BG", "LSN", "DEN", "PAT", "GRP"))
    BG = BG.astype(bool); log("loaded cached candidates", len(M))
else:
    zf = zipfile.ZipFile(ZIP)
    names = zf.namelist()
    ann_member = [n for n in names if n.endswith("annotations.json")][0]
    A = json.loads(zf.read(ann_member))
    img_member = {pathlib.Path(n).stem: n for n in names if n.lower().endswith(".png")}
    multi = {k: v for k, v in A.items() if len(v) == 4}
    log("patches:", len(A), " four-reader:", len(multi))
    rng = np.random.RandomState(0)
    cand, crops_l = [], []

    def crop_at(im, x, y):
        H, W = im.shape[:2]
        cx, cy = int(x / 100 * W), int(y / 100 * H)
        x0, y0 = max(0, cx - CROP // 2), max(0, cy - CROP // 2)
        c = im[y0:y0 + CROP, x0:x0 + CROP]
        return c if c.shape[:2] == (CROP, CROP) else None

    for name, p in multi.items():
        if name not in img_member: continue
        im = np.array(Image.open(io.BytesIO(zf.read(img_member[name]))).convert("RGB"))
        pts, who, lbl = [], [], []
        for a, lst in p.items():
            for d in lst:
                pts.append((d["x"], d["y"])); who.append(a); lbl.append(d["keypointlabels"])
        if not pts: continue
        P = np.array(pts)
        lab = DBSCAN(eps=RADIUS, min_samples=1).fit_predict(P)
        ids = sorted(set(lab)); dense = int(len(ids) >= 25)
        m = re.match(r"^(.+)_(\d+)_(\d+)$", name)
        grp = f"{m.group(1)}_{m.group(2)}" if m else name
        centers = []
        for l in ids:
            idx = np.where(lab == l)[0]
            c = P[idx].mean(0); centers.append(c)
            cr = crop_at(im, c[0], c[1])
            if cr is None: continue
            marked = np.array([int(any(who[i] == a for i in idx)) for a in READERS])
            top = collections.Counter(lbl[i] for i in idx).most_common(1)[0][0]
            cand.append((name, grp, marked, int(top in LES), dense, 0)); crops_l.append(cr)
        cc = np.array(centers); got = 0
        for _ in range(200):
            if got >= 8: break
            x, y = rng.uniform(5, 95, 2)
            if np.min(np.hypot(cc[:, 0] - x, cc[:, 1] - y)) > 5:
                cr = crop_at(im, x, y)
                if cr is None: continue
                cand.append((name, grp, np.zeros(4, int), 0, dense, 1)); crops_l.append(cr); got += 1
    crops = np.stack(crops_l)
    PAT = np.array([c[0] for c in cand]); GRP = np.array([c[1] for c in cand])
    M = np.array([c[2] for c in cand]); LSN = np.array([c[3] for c in cand])
    DEN = np.array([c[4] for c in cand]); BG = np.array([c[5] for c in cand]).astype(bool)
    np.savez_compressed(cache, crops=crops, M=M, BG=BG, LSN=LSN, DEN=DEN, PAT=PAT, GRP=GRP)
    log("candidates:", len(M), " background:", int(BG.sum()))

# ------------------------------------------------------------------ embeddings
emb_cache = OUT / f"emb_{FEATS}.npy"
if emb_cache.exists():
    E = np.load(emb_cache); log("loaded cached embeddings", E.shape)
else:
    mu = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); sd = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if FEATS == "resnet18":
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity(); size = 112
    else:
        from transformers import AutoModel
        hf = AutoModel.from_pretrained("facebook/dinov2-small"); size = 224
        net = lambda x: hf(pixel_values=x).pooler_output
        hf.eval()
    if FEATS == "resnet18": net.eval()
    fs = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            b = torch.tensor(crops[i:i + 64]).permute(0, 3, 1, 2).float() / 255.
            b = torch.nn.functional.interpolate(b, size=size, mode="bilinear")
            fs.append(net((b - mu) / sd).numpy())
            if (i // 64) % 30 == 0: log("embedded", i, "of", len(crops))
    E = np.vstack(fs); np.save(emb_cache, E); log("embeddings", E.shape)

# ------------------------------------------------------------------ experiment
Y2 = (M.sum(1) >= 2) & ~BG
Y3 = (M.sum(1) >= 3) & ~BG
log("reference positives >=2:", int(Y2.sum()), " >=3:", int(Y3.sum()), " lesion(>=2):", int((Y2 & (LSN == 1)).sum()))

def fit_linear(X, mode, s=None, e=None, y=None, steps=300, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=wd)
    if mode in ("naive", "oracle"):
        t = torch.tensor(s if mode == "naive" else y, dtype=torch.float32)
    else:
        st = torch.tensor(s, dtype=torch.float32); et = torch.tensor(e, dtype=torch.float32).clamp(min=0.05)
    for _ in range(steps):
        z = Xt @ w + b
        lp, ln = torch.nn.functional.softplus(-z), torch.nn.functional.softplus(z)
        if mode in ("naive", "oracle"):
            loss = (t * lp + (1 - t) * ln).mean()
        else:
            Rp = (st * lp / et).mean(); Rn = ln.mean() - (st * ln / et).mean()
            loss = Rp + torch.where(Rn >= 0, Rn, -0.5 * Rn)
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().numpy(), b.item()

def rates(tr):
    out = {"const": 0.5, "reader": np.zeros(4), "rcd": np.zeros((4, 2, 2))}
    num = den = 0
    for r in range(4):
        oth = (np.delete(M[tr], r, axis=1).sum(1) >= 2) & ~BG[tr]
        hit = M[tr, r] == 1
        out["reader"][r] = hit[oth].mean() if oth.sum() else 0.5
        num += hit[oth].sum(); den += oth.sum()
        for l in (0, 1):
            for d in (0, 1):
                m_ = oth & (LSN[tr] == l) & (DEN[tr] == d)
                out["rcd"][r, l, d] = hit[m_].mean() if m_.sum() >= 20 else out["reader"][r]
    out["const"] = num / max(den, 1)
    return out

METHODS = ["naive", "pu_const", "pu_reader", "pu_pred", "pu_true*", "oracle"]
SEEDS = (0, 1, 2)
store = {s: {k: np.zeros(len(E)) for k in METHODS} for s in SEEDS}
for fold, (tr, te) in enumerate(GroupKFold(5).split(E, Y2, GRP)):
    mu_, sd_ = E[tr].mean(0), E[tr].std(0) + 1e-6
    Xtr, Xte = (E[tr] - mu_) / sd_, (E[te] - mu_) / sd_
    rt = rates(tr)
    for seed in SEEDS:
        rg = np.random.RandomState(100 * fold + seed)
        pats = np.unique(PAT[tr]); rd = {p: rg.randint(4) for p in pats}
        r_i = np.array([rd[p] for p in PAT[tr]])
        s = np.where(BG[tr], 0, M[tr][np.arange(len(tr)), r_i]).astype(float)
        cnt = {p: int(s[PAT[tr] == p].sum()) for p in pats}
        c_i = np.array([cnt[p] for p in PAT[tr]])
        med = {r: np.median([cnt[p] for p in pats if rd[p] == r]) for r in range(4)}
        dproxy = np.array([int(c_i[i] > med[r_i[i]]) for i in range(len(tr))])
        li = np.where(s == 1)[0]
        cls = LogisticRegression(max_iter=1000, C=0.1).fit(Xtr[li], LSN[tr][li])
        pcls = (cls.predict_proba(Xtr)[:, 1] > 0.5).astype(int)
        es = {"pu_const": np.full(len(tr), rt["const"]), "pu_reader": rt["reader"][r_i],
              "pu_pred": rt["rcd"][r_i, pcls, dproxy], "pu_true*": rt["rcd"][r_i, LSN[tr], DEN[tr]]}
        for k in METHODS:
            if k == "naive": w, b = fit_linear(Xtr, "naive", s=s, seed=seed)
            elif k == "oracle": w, b = fit_linear(Xtr, "oracle", y=Y2[tr].astype(float), seed=seed)
            else: w, b = fit_linear(Xtr, "pu", s=s, e=es[k], seed=seed)
            store[seed][k][te] = Xte @ w + b
    log(f"fold {fold} done")

les_mask = ((LSN == 1) & ~BG) | BG
slides = np.unique(GRP)
by_slide = {g: np.where((GRP == g) & les_mask)[0] for g in slides}

def m3(sc, idx, yl):
    yy, ss = yl[idx], sc[idx]
    if yy.sum() < 5 or yy.sum() == len(yy): return np.full(3, np.nan)
    fpr, tpr, _ = roc_curve(yy, ss)
    return np.array([roc_auc_score(yy, ss), average_precision_score(yy, ss), np.interp(0.05, fpr, tpr)])

lines = []
def emit(s=""): print(s, flush=True); lines.append(s)

for nm, Yr in (("reference = marked by >=2 readers", Y2), ("reference = marked by >=3 readers", Y3)):
    yl = Yr & (LSN == 1)
    emit(f"\n=== {nm}  (features: {FEATS}) ===")
    emit(f"{'method':14s} {'all AUROC':>9s} {'all AP':>7s} | {'les AUROC':>9s} {'les AP':>7s} {'les rec@5%':>10s}")
    allm = np.ones(len(E), bool); les_idx = np.where(les_mask)[0]
    for k in METHODS:
        a = np.nanmean([m3(store[s][k], np.arange(len(E)), Yr)[:2] if False else
                        [roc_auc_score(Yr, store[s][k]), average_precision_score(Yr, store[s][k])] for s in SEEDS], axis=0)
        l = np.nanmean([m3(store[s][k], les_idx, yl) for s in SEEDS], axis=0)
        emit(f"{k:14s} {a[0]:9.3f} {a[1]:7.3f} | {l[0]:9.3f} {l[1]:7.3f} {l[2]:10.3f}")
    rg = np.random.RandomState(0); d = {k: [] for k in ("pu_pred", "pu_reader", "pu_const")}
    for _ in range(300):
        idx = np.concatenate([by_slide[g] for g in rg.choice(slides, size=len(slides), replace=True)])
        base = [m3(store[s]["naive"], idx, yl) for s in SEEDS]
        for k in d:
            d[k].append(np.nanmean([m3(store[s][k], idx, yl) - base[j] for j, s in enumerate(SEEDS)], axis=0))
    for k in d:
        arr = np.array(d[k])
        emit(f"  {k} - naive (lesion):  " + "   ".join(
            f"{lab} {np.nanmean(arr[:, j]):+.3f} [{np.nanpercentile(arr[:, j], 2.5):+.3f},{np.nanpercentile(arr[:, j], 97.5):+.3f}]"
            for j, lab in enumerate(("AUROC", "AP", "rec@5%"))))

# ------------------------------------------------------------------ deferral
Tgt = np.where(BG, 0.0, M.sum(1) / 4.0)
q = np.zeros(len(E))
for fold, (tr, te) in enumerate(GroupKFold(5).split(E, Y2, GRP)):
    mu_, sd_ = E[tr].mean(0), E[tr].std(0) + 1e-6
    q[te] = np.clip(Ridge(alpha=50.0).fit((E[tr] - mu_) / sd_, Tgt[tr]).predict((E[te] - mu_) / sd_), 0, 1)
emit(f"\n=== deferral (reference >=2 readers): disagreement-head corr = {np.corrcoef(q, Tgt)[0, 1]:.3f} ===")
rank = lambda a: rankdata(a) / len(a)
def amb_signals(sc, t, idx):
    ac = -np.abs(sc[idx] - t) / (np.std(sc[idx]) + 1e-9)
    ad = 1 - np.abs(2 * q[idx] - 1)
    return {"confidence": ac, "predicted disagreement": ad, "both (mean rank)": (rank(ac) + rank(ad)) / 2}
def thr(sc):
    ts = np.quantile(sc, np.linspace(0.05, 0.95, 91))
    return ts[np.argmax([((sc > x) == Y2).mean() for x in ts])]
covs = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
tab = collections.defaultdict(list)
for s in SEEDS:
    sc = store[s]["pu_pred"]; t = thr(sc); idx = np.arange(len(sc))
    err = ((sc > t) != Y2).astype(float)
    sig = amb_signals(sc, t, idx); sig["random"] = np.random.RandomState(s).rand(len(sc))
    for nme, amb in sig.items():
        o = np.argsort(amb)
        tab[nme].append([err[o[:int(len(err) * c)]].mean() for c in covs])
emit(f"{'signal':26s}" + "".join(f"{c:>8.1f}" for c in covs))
for nme, arr in tab.items(): emit(f"{nme:26s}" + "".join(f"{v:8.3f}" for v in np.mean(arr, axis=0)))
rg = np.random.RandomState(1); all_by = {g: np.where(GRP == g)[0] for g in slides}
dd = {c: [] for c in (0.5, 0.7, 0.9)}
for _ in range(300):
    idx = np.concatenate([all_by[g] for g in rg.choice(slides, size=len(slides), replace=True)])
    for c in dd:
        v = []
        for s in SEEDS:
            sc = store[s]["pu_pred"]; t = thr(sc)
            err = ((sc[idx] > t) != Y2[idx]).astype(float)
            sg = amb_signals(sc, t, idx)
            k = int(len(idx) * c)
            v.append(err[np.argsort(sg["both (mean rank)"])[:k]].mean() - err[np.argsort(sg["confidence"])[:k]].mean())
        dd[c].append(np.mean(v))
emit("both minus confidence-only, error among accepted (negative = better):")
for c, v in dd.items():
    emit(f"  coverage {c}: {np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")
(OUT / f"results_{FEATS}.txt").write_text("\n".join(lines) + "\n")
log("done; wrote", OUT / f"results_{FEATS}.txt")
