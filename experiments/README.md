# Experiments

Things that were measured but are not part of the deployed service.

## Scripts

Each runs from the repo root and writes COCO prediction dumps or its own output directory; none of them is imported by `app.py`.

| Script | What it does | Why it is not deployed |
|---|---|---|
| `slice_inference.py` | Runs the same weights on the full image and on overlapping 800×800 native-scale crops, one dump per view group | About 10 forward passes per image, and fused AP50 fell below the baseline |
| `ensemble_wbf.py` | Weighted box fusion over two members' dumps, then the same evaluation protocol | Fusion cost is the sum of its members and it did not improve detection score |
| `cascade_refine.py` | ConvNeXt-Tiny crop classifier that re-scores or re-labels detected boxes, trained only on train-split GT crops | Every variant it touched scored lower than the baseline on the 50 test images |

All three are covered by `tests/test_slice_inference.py`, `tests/test_ensemble.py` and `tests/test_cascade.py`; pytest finds them through the `experiments` entry in `pyproject.toml`.

The commands and the measured rows are in README §8 and `RESEARCH_BENCHMARKS.md` §6–7.

## Notebooks kept for provenance

These two notebooks are run recipes, not executed evidence:

- `notebooks/kaggle_benchmark_notebook.ipynb` — an earlier architecture comparison (Faster R-CNN with and without TTA, RetinaNet, fusion). It has no saved outputs and is superseded by `kaggle_beat_benchmark_notebook.ipynb` at the repo root.
- `notebooks/kaggle_cbct_scan2_notebook.ipynb` — the recipe used to produce the PreDentalSurgery segmentation. It has no saved outputs, so the Pre bundle's `run_metrics.json` is not used as evidence (README §7).

The executed evidence notebooks stay at the repo root: `kaggle_notebook.ipynb`, `kaggle_beat_benchmark_notebook.ipynb`, and `kaggle_cbct_notebook.ipynb`.

`kaggle_beat_benchmark_notebook.ipynb` calls these three scripts by their root-level names because that is how they were laid out when the run happened (repo commit `1ac1080`). The saved outputs are left exactly as executed; add the `experiments/` prefix to reproduce them now.
