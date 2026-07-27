"""CLI: explain a prediction as a per-window emotion timeline.

Usage:
    python scripts/explain.py --audio clip.wav
    python scripts/explain.py --audio clip.wav --model distilhubert-svm-synthetic:v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotionsense.inference.explain import explain_from_registry
from emotionsense.registry.local_registry import LocalRegistry


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-window explainability for one clip")
    ap.add_argument("--audio", required=True, help="path to an audio file")
    ap.add_argument(
        "--model", default=None, help="registry ref name:version (default: production default)"
    )
    ap.add_argument("--registry", default="experiments/artifacts/registry")
    args = ap.parse_args()

    registry = LocalRegistry(Path(args.registry))
    expl = explain_from_registry(args.audio, registry, model_ref=args.model)

    print(f"\nPrediction: {expl.predicted_label}  (confidence {expl.confidence:.3f})")
    print(f"Clip {expl.duration_sec:.2f}s split into {expl.n_windows} window(s)\n")
    print(f"{'window':>8} | {'span (s)':>12} | {'emotion':>9} | conf")
    print("-" * 46)
    for w in expl.windows:
        span = f"{w.start_sec:>5.2f}-{w.end_sec:<5.2f}"
        print(f"{w.index:>8} | {span} | {w.label:>9} | {w.confidence:.3f}")
    print(f"\nNote: {expl.note}")


if __name__ == "__main__":
    main()
