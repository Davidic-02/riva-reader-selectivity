# Reader selectivity in multi-pathologist cervical cytology annotations (RIVA)

Code and results for the manuscript *"Reader selectivity in multi-pathologist cervical cytology annotations: an analysis of RIVA and a negative result for reader-aware correction"* (David Adekoya; co-authors and affiliations to be added).

RIVA is a public set of 959 conventional Pap-smear patches in which 386 patches were marked independently by four cytopathologists and 573 by one of the same four. This repository measures how those readers differ in which cells they mark, tests whether treating unmarked cells as unlabelled (a propensity-weighted positive-unlabelled correction) helps a detector trained on single-reader labels, and tests a reader-disagreement head as a deferral signal.

## Main findings

| | Result |
|---|---|
| Cells also marked by each reader (radius 1.85%), all cells | 69% / 60% / 51% / 41% (readers 1 to 4) |
| Same, lesion cells only | 45% / 58% / 29% / 19% |
| Solo marks scoring as cell-like (image check) | 47% to 70% vs 4% of background |
| Class-aware correction vs naive, lesion AP (ResNet18) | +0.019 [-0.003, +0.049], not distinguishable from zero |
| Same with DINOv2-small | +0.005 [-0.017, +0.028] |
| Per-reader rate only vs naive, lesion AP | -0.030 [-0.049, -0.010] (ResNet18); -0.027 [-0.049, -0.009] (DINOv2-small) |
| Deferral: combined signal minus confidence, error among accepted at 50% coverage | -0.016 [-0.022, -0.011] (ResNet18); -0.015 [-0.021, -0.010] (DINOv2-small) |

This is a negative result for the correction. The deferral reference is defined by the same four readers, so that gain is partly circular. See the manuscript's Limitations section.

## Reproducing

The dataset is **not** included. Download RIVA from Zenodo (DOI 10.5281/zenodo.17288879) and check the archive:

```bash
curl -L -o riva_1.0.zip "https://zenodo.org/records/17288879/files/riva_1.0.zip?download=1"
md5 riva_1.0.zip    # expect 89329be851bac81c7b13bde413ae6a6f
```

Then create the environment and run the analyses (the archive is read in place; nothing is extracted):

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python src/riva_pipeline.py riva_1.0.zip out --features resnet18   # candidates, embeddings, experiment, deferral
./.venv/bin/python src/riva_pipeline.py riva_1.0.zip out --features dino       # same with DINOv2-small
./.venv/bin/python src/reader_behaviour.py riva_1.0.zip --cache out --out results/reader_behaviour.txt
./.venv/bin/python src/mixture_model.py riva_1.0.zip --out results/mixture_model.txt
./.venv/bin/python src/make_figures.py results paper/figures
```

Runtime on a laptop CPU: about two minutes with ResNet18 and about six with DINOv2-small. The pipeline writes `out/results_<features>.txt`; copy those into `results/` before running `make_figures.py`.

## Layout

- `src/riva_pipeline.py`: cell matching, candidates, embeddings, simulated single-reader experiment, bootstrap intervals, deferral.
- `src/reader_behaviour.py`: reader marking rates by class and density, skipped cells, Chao2 estimate, image check.
- `src/mixture_model.py`: zero-truncated real/spurious mixture (diagnostic).
- `src/make_figures.py`: figures for the manuscript.
- `results/`: text outputs of the runs reported in the manuscript.
- `paper/`: manuscript files and figures.

## Caveats

- Intervals are slide-level bootstrap intervals with trained models held fixed, so they understate training variance. One interval changed sign between two machines (Colab and a laptop).
- Candidates come from the readers' own marks plus random background points; there is no cell proposer, and the 573 real single-reader patches are not used.
- There is no independent reference standard (no biopsy or HPV result); consensus is defined by the same readers.

## Licence

Code: MIT (see LICENSE). **Dataset: not redistributed.** RIVA's licence wording differs between sources (paper: CC BY-NC-ND 4.0; Zenodo record: CC BY 4.0; packaged file: CC BY-NC 4.0). Derived crops and embeddings are git-ignored. Cite the dataset paper: Perez Bianchi P, et al. RIVA: An image dataset of conventional Pap smear cytology with multiple independent annotations. Sci Data 2025;12:1991. doi:10.1038/s41597-025-06280-2
