"""Zero-truncated mixture of real and spurious cells (Section 5.2 of the manuscript).

Each matched cell is real or spurious. A reader marks a real cell with probability
sigmoid(a_r + b_r*lesion + c_r*dense) and a spurious one with sigmoid(d_r). Cells that no
reader marked are never observed, so each likelihood is conditioned on at least one mark.
The model has two cell types only, so it cannot represent a real cell that few readers
notice; it is reported as a diagnostic, not as an estimate of reader error.

Usage: python src/mixture_model.py <riva_1.0.zip> [--out results/mixture_model.txt]
"""
import argparse, collections, itertools, json, pathlib, zipfile
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
from scipy.special import expit

READERS = [f"annotator_{i}" for i in range(1, 5)]
LES = {"SCC", "HSIL", "ASCH", "LSIL", "ASCUS", "ASC-H", "ASC-US"}
RADIUS = 1.85
ap = argparse.ArgumentParser(); ap.add_argument("zip"); ap.add_argument("--out", default=None)
args = ap.parse_args()
lines = []
def emit(s=""): print(s, flush=True); lines.append(s)

zf = zipfile.ZipFile(args.zip)
A = json.loads(zf.read([n for n in zf.namelist() if n.endswith("annotations.json")][0]))
X, L, D = [], [], []
for name, p in A.items():
    if len(p) != 4: continue
    pts, who, lbl = [], [], []
    for a, lst in p.items():
        for d in lst:
            pts.append((d["x"], d["y"])); who.append(a); lbl.append(d["keypointlabels"])
    if not pts: continue
    lab = DBSCAN(eps=RADIUS, min_samples=1).fit_predict(np.array(pts))
    ids = sorted(set(lab)); dense = int(len(ids) >= 25)
    for l in ids:
        idx = np.where(lab == l)[0]
        marked = {who[i] for i in idx}
        top = collections.Counter(lbl[i] for i in idx).most_common(1)[0][0]
        X.append([int(a in marked) for a in READERS]); L.append(int(top in LES)); D.append(dense)
X, L, D = np.array(X), np.array(L), np.array(D)
emit(f"cells: {len(X)}  lesion share {L.mean():.3f}  dense share {D.mean():.3f}")

def comps(th, X, L, D):
    A_, B_, C_, F_, P_ = th[0:4], th[4:8], th[8:12], th[12:16], th[16:18]
    S = expit(A_[None, :] + B_[None, :] * L[:, None] + C_[None, :] * D[:, None])
    F = np.tile(expit(F_)[None, :], (len(X), 1))
    pat = lambda Pr: np.prod(np.where(X == 1, Pr, 1 - Pr), axis=1)
    L1 = pat(S) / (1 - np.prod(1 - S, axis=1))
    L0 = pat(F) / (1 - np.prod(1 - F, axis=1))
    pi = expit(P_[0] + P_[1] * L)
    return S, F, L1, L0, pi

def nll(th, X, L, D):
    _, _, L1, L0, pi = comps(th, X, L, D)
    return -np.sum(np.log(pi * L1 + (1 - pi) * L0 + 1e-300)) + 0.01 * np.sum(th ** 2)

th0 = np.concatenate([np.zeros(12), np.full(4, -1.5), [1.0, 0.0]])
res = minimize(nll, th0, args=(X, L, D), method="L-BFGS-B")
th = res.x; emit(f"converged: {res.success}  nll {res.fun:.1f}")
emit("\nP(reader marks a real cell): non-lesion/sparse | lesion/sparse | lesion/dense      P(spurious mark)")
for i, a in enumerate(READERS):
    emit(f"  {a}: {expit(th[i]):.2f} | {expit(th[i]+th[4+i]):.2f} | {expit(th[i]+th[4+i]+th[8+i]):.2f}      {expit(th[12+i]):.3f}")
S, F, L1, L0, pi = comps(th, X, L, D)
post = pi * L1 / (pi * L1 + (1 - pi) * L0)
emit("\nsolo marks and mean posterior probability that they are real cells:")
for i, a in enumerate(READERS):
    solo = (X.sum(1) == 1) & (X[:, i] == 1)
    emit(f"  {a}: {solo.sum():4d} solo marks, mean P(real) {post[solo].mean():.2f}")
obs = np.bincount(X.sum(1), minlength=5); exp = np.zeros(5)
for pt in itertools.product([0, 1], repeat=4):
    if sum(pt) == 0: continue
    _, _, l1, l0, p_ = comps(th, np.tile(pt, (len(X), 1)), L, D)
    exp[sum(pt)] += np.sum(p_ * l1 + (1 - p_) * l0)
emit("\nreaders per cell: observed vs model-expected")
for k in range(1, 5): emit(f"  {k}: {obs[k]:5d} vs {exp[k]:7.1f}")
if args.out:
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n")
