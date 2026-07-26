"""Seed the serving environment: train + register a default model.

For the Lean Core runnable path this seeds the filesystem registry (no DB needed). In the
DB-backed production path this also inserts the admin user and dataset rows.

Usage:
    python scripts/seed_db.py
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    print("Seeding: training + registering svm-mfcc on synthetic as production default...")
    subprocess.run(
        [
            sys.executable,
            "scripts/train_and_register.py",
            "--model",
            "svm",
            "--dataset",
            "synthetic",
            "--promote",
        ],
        check=True,
    )
    print("Done. Default model registered. Start the API: uvicorn emotionsense.backend.main:app")


if __name__ == "__main__":
    main()
