"""Render leaderboard rows to a Markdown table + JSON export."""

from __future__ import annotations

import json
from pathlib import Path

from emotionsense.training.benchmark import LeaderboardRow, leaderboard_to_dicts


def _confusion_table(labels: list[str], matrix: list[list[int]]) -> list[str]:
    """Render a confusion matrix as a markdown table (rows=true, cols=pred)."""
    if not labels or not matrix:
        return ["_(no confusion matrix)_", ""]
    header = "| true \\ pred | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    out = [header, sep]
    for lab, row in zip(labels, matrix, strict=True):
        out.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    out.append("")
    return out


def to_markdown(rows: list[LeaderboardRow], title: str = "Benchmark Leaderboard") -> str:
    dicts = leaderboard_to_dicts(rows)
    lines = [
        f"# {title}",
        "",
        "UA = Unweighted Accuracy (headline, imbalance-robust). ± = std across CV folds.",
        "Times: Extract = audio→features per fold (0 if cached); Train = classifier fit; "
        "Infer = ms per test sample.",
        "",
        "| Model | Dataset | Eval | UA | Accuracy | macro-F1 | weighted-F1 | folds "
        "| Extract(s) | Train(s) | Infer(ms) |",
        "|-------|---------|------|----|----------|----------|-------------|-------"
        "|------------|----------|-----------|",
    ]
    for r in dicts:
        ua = f"{r['ua_mean']:.3f} ± {r['ua_std']:.3f}"
        acc = f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}"
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['eval_kind']} | {ua} | {acc} "
            f"| {r['macro_f1_mean']:.3f} | {r.get('weighted_f1_mean', 0.0):.3f} "
            f"| {r['n_folds']} | {r.get('extract_time_s', 0.0):.2f} "
            f"| {r.get('train_time_s', 0.0):.3f} | {r.get('infer_ms_per_sample', 0.0):.3f} |"
        )

    # Per-model confusion matrices.
    lines += ["", "## Confusion matrices", ""]
    for r in dicts:
        lines.append(f"### {r['model']} — {r['dataset']} ({r['eval_kind']})")
        lines += _confusion_table(r.get("confusion_labels", []), r.get("confusion", []))
    return "\n".join(lines) + "\n"


def export(
    rows: list[LeaderboardRow], out_dir: Path, name: str = "leaderboard"
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{name}.md"
    json_path = out_dir / f"{name}.json"
    md_path.write_text(to_markdown(rows), encoding="utf-8")
    json_path.write_text(json.dumps(leaderboard_to_dicts(rows), indent=2), encoding="utf-8")
    return md_path, json_path
