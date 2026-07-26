"""Render leaderboard rows to a Markdown table + JSON export."""

from __future__ import annotations

import json
from pathlib import Path

from emotionsense.training.benchmark import LeaderboardRow, leaderboard_to_dicts


def to_markdown(rows: list[LeaderboardRow], title: str = "Benchmark Leaderboard") -> str:
    dicts = leaderboard_to_dicts(rows)
    lines = [
        f"# {title}",
        "",
        "UA = Unweighted Accuracy (headline, imbalance-robust). ± = std across CV folds.",
        "",
        "| Model | Dataset | Eval | UA | Accuracy | macro-F1 | folds |",
        "|-------|---------|------|----|----------|----------|-------|",
    ]
    for r in dicts:
        ua = f"{r['ua_mean']:.3f} ± {r['ua_std']:.3f}"
        acc = f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}"
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['eval_kind']} | {ua} | {acc} "
            f"| {r['macro_f1_mean']:.3f} | {r['n_folds']} |"
        )
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
