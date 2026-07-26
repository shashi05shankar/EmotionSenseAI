# Notebooks — Colab / Kaggle training

The training plane runs on **free-tier GPU** (Colab/Kaggle). These notebooks are *thin* —
they install the package and call `emotionsense.*`, so logic never forks from `src/`.

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
