# Notebooks — Colab / Kaggle training

The training plane runs on **free-tier GPU** (Colab/Kaggle). These notebooks are *thin* —
they install the package and call `emotionsense.*`, so logic never forks from `src/`.

## RC1 experimental validation (current milestone)

Ready-to-run notebooks that validate the **frozen RC1** (`v1.0.0-rc1`) on real corpora
without any source changes — they install the tagged release and run the prepared
`configs/experiments/rc1_*.yaml`:

| Notebook | Platform | Datasets via |
|----------|----------|--------------|
| `kaggle_rc1_validation.ipynb` | Kaggle | attach Kaggle datasets, symlink into `data/raw/` |
| `colab_rc1_validation.ipynb`  | Colab  | `scripts/fetch_datasets.py` (RAVDESS direct + TESS/CREMA-D via Kaggle API) |

Experiments run: `rc1_validation` (RAVDESS 5-fold + cross-corpus CREMA-D), `rc1_cremad`
(CREMA-D in-corpus CV), `rc1_transformer` (Distil-HuBERT — SSL extraction is CPU-bound in
RC1). **Wait for these results before proposing any architectural change.**

---


| Notebook | Purpose |
|----------|---------|
| `01_download_data.ipynb` | Fetch + extract corpora into `data/raw/` |
| `02_train_classical.ipynb` | Baselines + classical models (CPU-ok) |
| `03_train_transformer_kaggle.ipynb` | Distil-HuBERT frozen-embeddings + head; **checkpoint/resume** to object storage so a killed Kaggle session resumes |
| `04_benchmark_leaderboard.ipynb` | Run the harness, export the leaderboard, log to MLflow |

## Minimal Colab/Kaggle cell

```python
!pip install -q "emotionsense[transformer] @ git+https://github.com/<you>/EmotionSenseAI"
from pathlib import Path
from emotionsense.common.yaml_config import load_experiment
from emotionsense.training.runner import run_experiment
from emotionsense.training.report import export

resolved = load_experiment("configs/experiments/baseline_suite.yaml")
rows = run_experiment(resolved, cache_dir=Path("data/features"))
export(rows, Path("experiments/reports"), name=resolved["name"])
```

### Why checkpoint/resume matters
Kaggle sessions time out (~9–12h). The transformer trainer writes checkpoints to object
storage every N steps; on restart it resumes from the latest checkpoint rather than
retraining from scratch. Prefer **Distil-HuBERT** over wav2vec2-large — near-equal accuracy
at a fraction of the compute (see `docs/research/02-sota-datasets-reference.md`).
